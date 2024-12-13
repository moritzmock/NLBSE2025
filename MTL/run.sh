#!/bin/bash

#################################
# Script to run the experiments #
#################################


# Function to kill all Python processes for a specific user
function cleanup {
    echo "Stopping all Python processes for user $(whoami)..."
    pkill -u $(whoami) python
}

# Trap SIGINT (Ctrl+C) signal
trap cleanup SIGINT

# set env
source env/bin/activate

# set pythonpath
export PYTHONPATH=$(pwd)


echo $(python <<EOF
import sys
if sys.prefix == sys.base_prefix:
        print("No, you are not in a virtual environment.")
else:
        print("Yes, you are in a virtual environment.")
EOF
)

nvidia-smi

python MTL/main.py --output-path ./results --model "roberta-base" --hs True --weight-method-name 'famo' --jobID-manual=rb 
python MTL/main.py --output-path ./results --model='distilroberta-base' --hs True --weight-method-name 'famo' --jobID-manual=drb 
python MTL/main.py --output-path ./results --model "microsoft/codebert-base" --hs True --weight-method-name 'famo' --jobID-manual=codbert 
python MTL/main.py --output-path ./results --model "microsoft/unixcoder-base" --hs True --weight-method-name 'famo' --jobID-manual=unixcoder 