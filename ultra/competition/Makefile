ULTRA_COMPETITION_DIR := $(shell pwd)

competition_bundle.zip: scoring_program.zip track1_evaluation_scenarios.zip track2_evaluation_scenarios.zip
	mkdir temp_competition_bundle/
	cp -r competition_bundle/* temp_competition_bundle/
	rm -r temp_competition_bundle/scoring_program/
	rm -r temp_competition_bundle/track1_evaluation_scenarios/
	rm -r temp_competition_bundle/track2_evaluation_scenarios/
	mv scoring_program.zip temp_competition_bundle/
	mv track1_evaluation_scenarios.zip temp_competition_bundle/
	mv track2_evaluation_scenarios.zip temp_competition_bundle/
	cd temp_competition_bundle/ && \
	zip -r $(ULTRA_COMPETITION_DIR)/competition_bundle.zip . && \
	cd -
	rm -r temp_competition_bundle/

track1_evaluation_scenarios.zip:
	cd competition_bundle/track1_evaluation_scenarios && \
	zip -r $(ULTRA_COMPETITION_DIR)/track1_evaluation_scenarios.zip . && \
	cd -

track2_evaluation_scenarios.zip:
	cd competition_bundle/track2_evaluation_scenarios && \
	zip -r $(ULTRA_COMPETITION_DIR)/track2_evaluation_scenarios.zip . && \
	cd -

scoring_program.zip:
	cd competition_bundle/scoring_program && \
	zip -r $(ULTRA_COMPETITION_DIR)/scoring_program.zip . && \
	cd -

starting_kit.zip:
	cd starting_kit && \
	zip -r $(ULTRA_COMPETITION_DIR)/starting_kit.zip . -x '*__pycache__*' 'scenarios/pool/**/*.glb' && \
	cd -
