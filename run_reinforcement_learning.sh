NUM_ACTORS=$1
GPU_USE=$2
ENVIRONMNET=$3
MODEL_NAME=$4
GRADIENT_CLIPPING=$5

tmux new-session -d -t impala_pysc2

tmux new-window -d -n learner
COMMAND_LEARNER='python3.8 learner.py --env_num '"${NUM_ACTORS}"' --gpu_use '"${GPU_USE}"' --model_name '"${MODEL_NAME}"'  --gradient_clipping '"${GRADIENT_CLIPPING}"''
echo $COMMAND_LEARNER

tmux send-keys -t "learner" "$COMMAND_LEARNER" ENTER

sleep 8.0

for ((id=0; id < $NUM_ACTORS; id++)); do
    tmux new-window -d -n "actor_${id}"
    COMMAND='python3.8 actor.py --env_id '"${id}"' --environment '"${ENVIRONMNET}"''
    tmux send-keys -t "actor_${id}" "$COMMAND" ENTER

    sleep 2.0
done

tmux attach -t impala_pysc2
