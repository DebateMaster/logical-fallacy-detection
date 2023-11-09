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
* `mlpt/datamodules` - Dataset generation and data augmentation modules like in pytorch datasets.
* `mlpt/models` - Model definition and training.
* `mlpt/utils` - Utils for training and inference.
* `mlpt/modules` - Pytorch network modules.