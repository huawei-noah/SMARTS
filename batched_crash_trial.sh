OUTPUT_DIR=$1
CPUS=$2
TOTAL_COUNT=$3
MAX_STEPS_PER_TRIAL=$4

if (( $CPUS < 1 )); then
    CPUS=1
fi

CURRENT=30
LOOPS=$(( $TOTAL_COUNT / $CPUS ))
for (( i=0; i<$LOOPS; i++ ))
do
    base=$(( $i * $CPUS ))
    echo iter: $i
    for (( j=0; j<$CPUS; j++ ))
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
    for (( j=0; j<$CPUS; j++ ))
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
