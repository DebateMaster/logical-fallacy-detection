# Logical Fallacy Detection
A repository containing the network to detect and classify logical fallacies in text.

# Model evaluation
## Binary classification

![bin train accuracy](etc/img/bin_train_acc.png "Roberta training accuracy") ![bin train loss](etc/img/bin_train_loss.png "Roberta training loss")
![bin eval accuracy](etc/img/bin_val_acc.png "Roberta validation accuracy") ![bin eval loss](etc/img/bin_val_loss.png "Roberta validation loss")

## Multiclass classification
![electra train accuracy](etc/img/electra_train_acc.png "Electra training accuracy") ![electra train loss](etc/img/electra_loss.png "Electra training loss")
![electra eval accuracy](etc/img/electra_eval_accuracy.png "Electra validation accuracy") ![electra eval loss](etc/img/electra_val_loss.png "Electra validation loss")



* Run docker 
* Modify docker settings to: 
```json 
{
  "builder": {
    "gc": {
      "defaultKeepStorage": "20GB",
      "enabled": true
    }
  },
  "experimental": false,
  "features": {
    "buildkit": false
  }
}
```

* Build and run docker
```bash
chmod +x deploy.sh
./deploy.sh
```


### Repo structure
* `network/models` - Models weights.
* `network/utils` - Models definitions and utils for inference.
* `network/training/data` - Data used for training binary and multiclass classification models.
* `network/training/baseline` - Code for training baseline NLI models.
* `network/training/prototex` - Code for training CBR ProtoTEx model.
