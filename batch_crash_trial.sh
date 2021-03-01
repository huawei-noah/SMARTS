for TRIAL in 0 1 2 3
do
    echo "Trial_${TRIAL} starting"
    bash ./crash_trial.sh ./replay_${TRIAL} &
    echo "Trial_${TRIAL} ended"
done