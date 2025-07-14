# Loading datasets in the memory

```
    import torch
    t = torch.load('1_eval_data.tar.pth')
    print(t.keys()) # it will print data and targets
    data, targets = t['data'], t['targets'] # both numpy.ndarray
```