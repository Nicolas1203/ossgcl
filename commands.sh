# Offline training
python main.py --learner CE --batch-size 256 --training-type uni --epoch 50 --dataset cifar10 --n-runs 2 -t offline,cifar10
python main.py --learner CE --batch-size 256 --training-type uni --epoch 50 --dataset cifar100 --n-runs 2 -t offline,cifar100

# Fine tuning
python main.py --learner CE --batch-size 256 --training-type inc --dataset cifar10 --n-runs 2 -t finetune,cifar10
python main.py --learner CE --batch-size 256 --training-type inc --dataset cifar100 --n-runs 2 - -t finetune,cifar100

# Experience replay
## cifar10
python main.py --learner ER --buffer reservoir --mem-size 200 --mem-batch-size 100 --batch-size 10 --dataset cifar10 --n-runs 2 -t er,cifar10,m200
python main.py --learner ER --buffer reservoir --mem-size 500 --mem-batch-size 100 --batch-size 10 --dataset cifar10 --n-runs 2 -t er,cifar10,m500
## cifar100
python main.py --learner ER --buffer reservoir --mem-size 2000 --mem-batch-size 500 --batch-size 10 --dataset cifar100 --n-runs 2 -t er,cifar100,m2000
python main.py --learner ER --buffer reservoir --mem-size 5000 --mem-batch-size 500 --batch-size 10 --dataset cifar100 --n-runs 2 -t er,cifar100,m5000

# Supervised Contrastive Replay
## cifar10
python main.py --learner SCR --buffer reservoir --mem-size 200 --mem-batch-size 100 --batch-size 10 --dataset cifar10 --n-runs 2 -t scr,cifar10,m200
python main.py --learner SCR --buffer reservoir --mem-size 500 --mem-batch-size 100 --batch-size 10 --dataset cifar10 --n-runs 2 -t scr,cifar10,m500
## cifar100
python main.py --learner SCR --buffer reservoir --mem-size 2000 --mem-batch-size 500 --batch-size 10 --dataset cifar100 --n-runs 2 -t scr,cifar100,m2000
python main.py --learner SCR --buffer reservoir --mem-size 5000 --mem-batch-size 500 --batch-size 10 --dataset cifar100 --n-runs 2 -t scr,cifar100,m5000

# Experience replay - Memory Only
## cifar10
python main.py --learner ER --memory-only --buffer reservoir --mem-size 200 --mem-batch-size 100 --batch-size 10 --dataset cifar10 --n-runs 2 -t er,mo,cifar10,m200
python main.py --learner ER --memory-only --buffer reservoir --mem-size 500 --mem-batch-size 100 --batch-size 10 --dataset cifar10 --n-runs 2 -t er,mo,cifar10,m500
## cifar100
python main.py --learner ER --memory-only --buffer reservoir --mem-size 2000 --mem-batch-size 500 --batch-size 10 --dataset cifar100 --n-runs 2 -t er,mo,cifar100,m2000
python main.py --learner ER --memory-only --buffer reservoir --mem-size 5000 --mem-batch-size 500 --batch-size 10 --dataset cifar100 --n-runs 2 -t er,mo,cifar100,m5000

# Supervised Contrastive Replay - Memory Only
## cifar10
python main.py --learner SCR --memory-only --buffer reservoir --mem-size 200 --mem-batch-size 100 --batch-size 10 --dataset cifar10 --n-runs 2 -t scr,mo,cifar10,m200
python main.py --learner SCR --memory-only --buffer reservoir --mem-size 500 --mem-batch-size 100 --batch-size 10 --dataset cifar10 --n-runs 2 -t scr,mo,cifar10,m500
## cifar100
python main.py --learner SCR --memory-only --buffer reservoir --mem-size 2000 --mem-batch-size 500 --batch-size 10 --dataset cifar100 --n-runs 2 -t scr,mo,cifar100,m2000
python main.py --learner SCR --memory-only --buffer reservoir --mem-size 5000 --mem-batch-size 500 --batch-size 10 --dataset cifar100 --n-runs 2 -t scr,mo,cifar100,m5000

# Semi-Supervised Contrastive Replay
## cifar10
python main.py --learner SSCL --buffer reservoir --mem-size 200 --mem-batch-size 100 --batch-size 10 --dataset cifar10 --n-runs 2 -t sscr,cifar10,m200
python main.py --learner SSCL --buffer reservoir --mem-size 500 --mem-batch-size 100 --batch-size 10 --dataset cifar10 --n-runs 2 -t sscr,cifar10,m500
## cifar100
python main.py --learner SSCL --buffer reservoir --mem-size 2000 --mem-batch-size 500 --batch-size 10 --dataset cifar100 --n-runs 2 -t sscr,cifar100,m2000
python main.py --learner SSCL --buffer reservoir --mem-size 5000 --mem-batch-size 500 --batch-size 10 --dataset cifar100 --n-runs 2 -t sscr,cifar100,m5000
