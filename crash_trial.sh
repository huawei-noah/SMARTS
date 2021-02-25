BEST_STEPS=9999
TEMP_DIR=${1}_TMP
for (( SPEED=6; SPEED<=500; SPEED++ ))
do
    rm -r $TEMP_DIR
    python3.7 examples/bugtest.py scenarios/loop --speed ${SPEED} --save-dir $TEMP_DIR --headless
    STEPS=$?
    if (( $STEPS < $BEST_STEPS )); then
        echo $STEPS > ${TEMP_DIR}/steps.txt
        echo $SPEED > ${TEMP_DIR}/speed.txt
        rm -r $1
        mv ${TEMP_DIR} $1
        BEST_STEPS=$STEPS
    fi
done