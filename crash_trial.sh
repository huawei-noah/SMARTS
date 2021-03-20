TEMP_DIR=${1}_TMP
START=${2}
COUNT=${3}
MAX_STEPS=${4}
BEST_STEPS=$MAX_STEPS


for (( SPEED=$START; SPEED<$START + $COUNT; SPEED++ ))
do
    if [[ -d $TEMP_DIR ]]; then
        rm -r $TEMP_DIR
    fi
    PYTHONHASHSEED=42 python3.7 examples/bugtest.py scenarios/loop --speed ${SPEED} --save-dir $TEMP_DIR --max-steps $MAX_STEPS --write --headless
    STEPS=$?
    if (( $STEPS < $BEST_STEPS )); then
        echo $STEPS > ${TEMP_DIR}/steps.txt
        echo $SPEED > ${TEMP_DIR}/speed.txt
        echo "Steps taken ${STEPS}0, Speed: ${SPEED}"
        if [[ -d $1 ]]; then
            rm -r $1
        fi
        mv ${TEMP_DIR} $1
        BEST_STEPS=$STEPS
        break
    fi
done
if [[ -d $TEMP_DIR ]]; then
    rm -r $TEMP_DIR
fi