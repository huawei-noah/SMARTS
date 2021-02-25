for TRIAL in 0 1 2 3 4
do
    bash ./crash_trial.sh ./replay_${TRIAL} &
done