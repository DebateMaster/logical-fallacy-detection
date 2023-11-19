# Logical Fallacy Detection
A repository containing the network to detect and classify logical fallacies in text.


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
chmod +x build_run.sh
./build_run.sh
```


### Repo structure
* `network/datamodules` - Dataset generation and data augmentation modules like in pytorch datasets.
* `network/models` - Model definition and training.
* `network/utils` - Utils for training and inference.
* `network/modules` - Pytorch network modules.