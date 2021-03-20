OUTPUT_DIR=$1
NUM_PARALLEL_TRIALS=$2
TOTAL_COUNT=$3
MAX_STEPS_PER_TRIAL=$4

if (( $NUM_PARALLEL_TRIALS < 1 )); then
    NUM_PARALLEL_TRIALS=1
fi

CURRENT=30
LOOPS=$(( $TOTAL_COUNT / $NUM_PARALLEL_TRIALS ))
for (( i=0; i<$LOOPS; i++ ))
do
    base=$(( $i * $NUM_PARALLEL_TRIALS ))
    echo iter: $i
    for (( j=0; j<$NUM_PARALLEL_TRIALS; j++ ))
    do
        TEMP_DIR="tmp_${j}/${OUTPUT_DIR}" 
        num=$(( $CURRENT + $j + $base ))
        bash crash_trial.sh $TEMP_DIR $num 1 $MAX_STEPS_PER_TRIAL &
        pids[${j}]=$!
    done

    for pid in ${pids[*]}
    do
        wait $pid
    done

    BEST=$MAX_STEPS_PER_TRIAL
    for (( j=0; j<$NUM_PARALLEL_TRIALS; j++ ))
    do
        TEMP_DIR="tmp_${j}/${OUTPUT_DIR}"
        STEP_FILE="${TEMP_DIR}/steps.txt"
        if [[ ! -f $STEP_FILE ]]; then
            continue
        fi
        STEPS=$(( $(cat ${STEP_FILE}) * 10 ))
        if (( $STEPS < $BEST )); then
            BEST=$STEPS
            rm -r ${OUTPUT_DIR}
            cp -r ${TEMP_DIR} ${OUTPUT_DIR}
        fi
        rm -r "tmp_${j}"
    done
    
    if (( $BEST < $MAX_STEPS_PER_TRIAL )); then
        break
    fi
done
