import wandb

def align_print_dic(dic, message):
    """Print dictionary with aligned keys and a header message."""
    print(f"-----------------------------{message}-----------------------------")
    max_key_length = max([len(str(key)) for key in dic.keys()])
    for key, value in dic.items():
        print(f"{key:{max_key_length}}: {value}")


#implement a function called print_or_wandb_log that takes wandb_switch, print_dic and message
def print_or_wandb_log(wandb_switch, print_dic, message):
    """Log to wandb if enabled, otherwise print dictionary with aligned formatting."""
    if wandb_switch:
        wandb.log(dict(print_dic))
    else:
        align_print_dic(print_dic, message)