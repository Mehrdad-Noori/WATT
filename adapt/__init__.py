from .watt import WATT


def get_method(args, device):
    if args.method == 'watt':
        print(f"Selected method: WATT with parameters: backbone={args.backbone}, lr={args.lr}, type={args.watt_type}, l={args.watt_l}, m={args.watt_m}, templates dir={args.watt_temps}, use reference template for evaluation={args.watt_reference_for_evaluation}, device={device}")
        return WATT(args.backbone, args.lr, type=args.watt_type, l=args.watt_l, m=args.watt_m, device=device)
    
    # Add other methods here 
    else:
        raise ValueError(f"Unknown method: {args.method}")