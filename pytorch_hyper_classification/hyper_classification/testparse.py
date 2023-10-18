
import argparse
parser = argparse.ArgumentParser(description='PyTorch Classification Training')
    
parser.add_argument('--demo', dest="user_attr.demo", help='shape to which images are resized for training: (h, w)')
args = parser.parse_args()
print(args.__dict__)
new_dict = args.__dict__
del new_dict["user_attr.demo"]
args.__dict__ = new_dict
print(args.__dict__)
