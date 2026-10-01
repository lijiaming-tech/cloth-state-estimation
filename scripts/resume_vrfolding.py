"""Compatibility entry point; an explicit --resume checkpoint is required."""
import sys
from train_vrfolding_from_scratch import main

if __name__ == '__main__':
    if not {'--help', '-h'}.intersection(sys.argv[1:]) and '--resume' not in sys.argv[1:] and not any(arg.startswith('--resume=') for arg in sys.argv[1:]):
        raise SystemExit('Use --resume /absolute/path/to/latest.pt; automatic selection was removed.')
    main()
