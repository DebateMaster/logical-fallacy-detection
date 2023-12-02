import torch
from transformers import ElectraForSequenceClassification
import numpy as np

class ProtoTEx_electra(torch.nn.Module):
    def __init__(
        self,
        num_prototypes,
        num_pos_prototypes,
        n_classes=14,
        class_weights=None,
        bias=True,
        dropout=False,
        special_classfn=False,
        p=0.5,
        batchnormlp1=False,
    ):
        super().__init__()
        self.electra_model = ElectraForSequenceClassification.from_pretrained("howey/electra-base-mnli").electra
        self.electra_out_dim = self.electra_model.config.hidden_size
        self.one_by_sqrt_electraoutdim = 1 / torch.sqrt(
            torch.tensor(self.electra_out_dim).float()
        )
        
        self.max_position_embeddings = 128
        self.num_protos = num_prototypes
        self.num_pos_protos = num_pos_prototypes
        self.num_neg_protos = self.num_protos - self.num_pos_protos

        self.pos_prototypes = torch.nn.Parameter(
            torch.rand(
                self.num_pos_protos, self.max_position_embeddings, self.electra_out_dim
            )
        )
        if self.num_neg_protos > 0:
            self.neg_prototypes = torch.nn.Parameter(
                torch.rand(
                    self.num_neg_protos, self.max_position_embeddings, self.electra_out_dim
                )
            )
        self.n_classes = n_classes
        # TODO: Try setting bias to True
        self.classfn_model = torch.nn.Linear(
            self.num_protos, n_classes, bias=bias
        )

        #         self.loss_fn=torch.nn.BCEWithLogitsLoss(reduction="mean")
        if class_weights is not None:
            print("Using class weights for cross entropy loss...")
            self.loss_fn = torch.nn.CrossEntropyLoss(
                weight=torch.Tensor(class_weights), reduction="mean"
            )
        else:
            self.loss_fn = torch.nn.CrossEntropyLoss(reduction="mean")

        self.do_dropout = dropout
        self.special_classfn = special_classfn
        self.dropout = torch.nn.Dropout(p=p)
        self.dobatchnorm = (
            batchnormlp1  ## This flag is actually for instance normalization
        )
        self.distance_grounder = torch.zeros(
            n_classes, self.num_protos
        ).cuda()
        for i in range(n_classes):
            # Approach 1: 50% Random initialization
            # self.distance_grounder[i][np.random.randint(0, self.num_protos, int(self.num_protos / 2))] = 1e7
            # Approach 2: Original Prototex paper approach
            if self.num_neg_protos > 0 and i == 0:
                self.distance_grounder[0][:self.num_pos_protos] = 1e7
            self.distance_grounder[i][self.num_pos_protos:] = 1e7
            # Approach 3: A mistake but works well
            # if self.num_neg_protos > 0 and i == 0:
            #     self.distance_grounder[0][self.num_pos_protos:] = 1e7
            # self.distance_grounder[i][:self.num_pos_protos] = 1e7
            # Approach 4: For the case that we want each class to be connected to at least k prototypes which is 3 in our case
            # self.distance_grounder[i][3*i:3*i + 3] = 1e7

    def set_prototypes(self, input_ids_pos_rdm, attn_mask_pos_rdm, do_random=False):
        print("initializing prototypes with xavier init")
        torch.nn.init.xavier_normal_(self.pos_prototypes)
        if self.num_neg_protos > 0:
            torch.nn.init.xavier_normal_(self.neg_prototypes)
    
    def set_encoder_status(self, status=True):
        self.num_enc_layers = len(self.electra_model.encoder.layer)

        for i in range(self.num_enc_layers):
            self.electra_model.encoder.layer[i].requires_grad_(False)
        self.electra_model.encoder.layer[
            self.num_enc_layers - 1
        ].requires_grad_(status)
        return

    def set_classfn_status(self, status=True):
        self.classfn_model.requires_grad_(status)

    def set_protos_status(self, pos_or_neg=None, status=True):
        if pos_or_neg == "pos" or pos_or_neg is None:
            self.pos_prototypes.requires_grad = status
        if self.num_neg_protos > 0:
            if pos_or_neg == "neg" or pos_or_neg is None:
                self.neg_prototypes.requires_grad = status

    def forward(
        self,
        input_ids,
        attn_mask,
        use_decoder=1,
        use_classfn=0,
        use_rc=0,
        use_p1=0,
        use_p2=0,
        use_p3=0,
        classfn_lamb=1.0,
        rc_loss_lamb=0.95,
        p1_lamb=0.93,
        p2_lamb=0.92,
        p3_lamb=1.0,
        distmask_lp1=0,
        distmask_lp2=0,
        pos_or_neg=None,
        random_mask_for_distanceMat=None,
    ):
        """
        1. p3_loss is the prototype-distance-maximising loss. See the set of lines after the line "if use_p3:"
        2. We also have flags distmask_lp1 and distmask_lp2 which uses "masked" distance matrix for calculating lp1 and lp2 loss.
        3. the flag "random_mask_for_distanceMat" is an experimental part. It randomly masks (artificially inflates)
        random places in the distance matrix so as to encourage more prototypes be "discovered" by the training
        examples.
        """
        batch_size = input_ids.size(0)
        
        last_hidden_state = self.electra_model(
            input_ids.cuda(),
            attn_mask.cuda(),
            output_attentions=False,
            output_hidden_states=False,
        ).last_hidden_state

        # Lp3 is minimize the negative of inter-prototype distances (maximize the distance)
        input_for_classfn, l_p1, l_p2, l_p3, l_p4, classfn_out, classfn_loss = (
            None,
            torch.tensor(0),
            torch.tensor(0),
            torch.tensor(0),
            torch.tensor(0),
            None,
            torch.tensor(0),
        )

        if self.num_neg_protos > 0:
            all_protos = torch.cat((self.neg_prototypes, self.pos_prototypes), dim=0)
        else:
            all_protos = self.pos_prototypes

        if use_classfn or use_p1 or use_p2:
            if not self.dobatchnorm:
                ## TODO: This loss function is not ignoring the padded part of the sequence; Get element-wise distane and then multiply with the mask
                input_for_classfn = torch.cdist(
                    last_hidden_state.view(batch_size, -1),
                    all_protos.view(self.num_protos, -1),
                )
            else:
                # TODO: Try cosine distance
                input_for_classfn = torch.cdist(
                    last_hidden_state.view(batch_size, -1),
                    all_protos.view(self.num_protos, -1),
                )
                input_for_classfn = torch.nn.functional.instance_norm(
                    input_for_classfn.view(batch_size, 1, self.num_protos)
                ).view(batch_size, self.num_protos)

        if use_classfn:
            if self.do_dropout:
                if self.special_classfn:
                    classfn_out = (
                        input_for_classfn @ self.classfn_model.weight.t()
                        + self.dropout(self.classfn_model.bias.repeat(batch_size, 1))
                    ).view(batch_size, self.n_classes)
                else:
                    classfn_out = self.classfn_model(
                        self.dropout(input_for_classfn)
                    ).view(batch_size, self.n_classes)
            else:
                classfn_out = self.classfn_model(input_for_classfn).view(
                    batch_size, self.n_classes)

        return classfn_out
