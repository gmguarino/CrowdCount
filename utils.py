import torchvision.transforms as T

def transforms(x, target=False):
    transform_list = [
        T.RandomHorizontalFlip(0.5)
    ]
    if not target:
        transform_list = [T.ToTensor()] + transform_list
    transform_function = T.Compose(transform_list)
    return transform_function(x)