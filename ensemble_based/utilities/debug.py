import wandb
import logging

def setup_logging():
    """Setup basic logging configuration"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

def log_info(message):
    """Log an info message"""
    logging.info(message)

def align_print_dic(dic, message):
    """Print dictionary with aligned keys and a header message."""
    print(f"-----------------------------{message}-----------------------------")
    max_key_length = max([len(str(key)) for key in dic.keys()])
    for key, value in dic.items():
        print(f"{key:{max_key_length}}: {value}")

def print_network_grad(network):
    """Print gradient information for all network parameters."""
    for name, param in network.named_parameters():
        if param.grad == None:
            print(name, param.grad)
        else:
            print(name, param.grad.shape)            

def print_with_dash(message):
    """Print message surrounded by dashes for emphasis."""
    print(f"-----------------------------{message}-----------------------------")

def print_or_wandb_log(wandb_switch, print_dic, message):
    """Log to wandb if enabled, otherwise print dictionary with aligned formatting."""
    if wandb_switch:
        wandb.log(dict(print_dic))
    else:
        align_print_dic(print_dic, message)
