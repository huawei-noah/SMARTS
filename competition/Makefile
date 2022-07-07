COMPETITION_DIR := $(shell pwd)

competition_bundle.zip: scoring_program.zip track1_evaluation.zip track2_evaluation.zip
	mkdir temp_competition_bundle/
	cp -r competition_bundle/* temp_competition_bundle/
	rm -r temp_competition_bundle/scoring_program/
	rm -r temp_competition_bundle/track1_evaluation/
	rm -r temp_competition_bundle/track2_evaluation/
	mv scoring_program.zip temp_competition_bundle/
	mv track1_evaluation.zip temp_competition_bundle/
	mv track2_evaluation.zip temp_competition_bundle/
	cd temp_competition_bundle/ && \
	zip -r $(COMPETITION_DIR)/competition_bundle.zip . && \
	cd -
	rm -r temp_competition_bundle/

track1_evaluation.zip:
	cd competition_bundle/track1_evaluation && \
	zip -r $(COMPETITION_DIR)/track1_evaluation.zip . && \
	cd -

track2_evaluation.zip:
	cd competition_bundle/track2_evaluation && \
	zip -r $(COMPETITION_DIR)/track2_evaluation.zip . && \
	cd -

scoring_program.zip:
	cp -r evaluation/* competition_bundle/scoring_program && \
	cd competition_bundle/scoring_program && \
	zip -r $(COMPETITION_DIR)/scoring_program.zip . && \
	cd - && \
	rm competition_bundle/scoring_program/*.py

starting_kit.zip:
	cp -r ./train ./submission ./starting_kit && \
	cd starting_kit && \
	zip -r $(COMPETITION_DIR)/starting_kit.zip . -x '*__pycache__*' 'scenarios/pool/**/*.glb' && \
	cd - && \
	rm -r ./starting_kit/*

submission.zip:
	cd submission && \
	zip -r $(COMPETITION_DIR)/submission.zip . -x '*__pycache__*' && \
	cd -