#!/bin/bash
cd "$(dirname "$0")/.."
python3 exp/bank.py train 180 11
python3 exp/bank.py test 40 13
python3 exp/bank.py val 12 12
python3 exp/bank.py ood 20 14
echo BANK_DONE
python3 exp/sweeps.py
echo SWEEPS_DONE
