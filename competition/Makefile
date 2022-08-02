COMPETITION_DIR := $(shell pwd)

competition_bundle.zip: scoring_program.zip validation_evaluation.zip track1_evaluation.zip track2_evaluation.zip
	mkdir temp_competition_bundle/
	cp -r competition_bundle/* temp_competition_bundle/
	rm -r temp_competition_bundle/scoring_program/
	rm -r temp_competition_bundle/validation_evaluation/
	rm -r temp_competition_bundle/track1_evaluation/
	rm -r temp_competition_bundle/track2_evaluation/
	mv scoring_program.zip temp_competition_bundle/
	mv validation_evaluation.zip temp_competition_bundle/
	mv track1_evaluation.zip temp_competition_bundle/
	mv track2_evaluation.zip temp_competition_bundle/
	cd temp_competition_bundle/ && \
	zip -r $(COMPETITION_DIR)/competition_bundle.zip . && \
	cd -
	rm -r temp_competition_bundle/

validation_evaluation.zip:
	cd competition_bundle/validation_evaluation && \
	zip -r $(COMPETITION_DIR)/validation_evaluation.zip . && \
	cd -

track1_evaluation.zip:
	mkdir temp_track1_evaluation/
	cp -r competition_bundle/track1_evaluation/* temp_track1_evaluation/
	cp -r eval_scenarios temp_track1_evaluation
	cd temp_track1_evaluation && \
	zip -r $(COMPETITION_DIR)/track1_evaluation.zip . && \
	cd -
	rm -r temp_track1_evaluation

track2_evaluation.zip:
	cd competition_bundle/track2_evaluation && \
	zip -r $(COMPETITION_DIR)/track2_evaluation.zip . && \
	cd -

scoring_program.zip:
	mkdir temp_scoring_program/
	cp -r competition_bundle/scoring_program/* temp_scoring_program/
	cp -r evaluation/* temp_scoring_program/
	cd temp_scoring_program && \
	zip -r $(COMPETITION_DIR)/scoring_program.zip . && \
	cd -
	rm -r temp_scoring_program

track1_submission.zip:
	cd track1/submission && \
	zip -r $(COMPETITION_DIR)/track1_submission.zip . -x '*__pycache__*' && \
	cd -

track2_submission.zip:
	cd track2 && \
	zip -r $(COMPETITION_DIR)/track2.zip . -x '*__pycache__*' && \
	cd -