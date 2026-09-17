.PHONY: env-up env-down env-cleanup cv-train cv-shell cv-demo

env-up:
	@docker compose up -d --build

env-down:
	@docker compose stop

env-cleanup:
	@docker compose down --remove-orphans

cv-train: env-up
	@docker compose exec cv python src/train.py

cv-shell: env-up
	@docker compose exec cv bash

cv-demo: env-up
	@docker compose exec cv python src/predict.py /clothes_cv/input/$(IMAGE)