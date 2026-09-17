FROM ubuntu:latest
LABEL authors="nekaido"

ENTRYPOINT ["top", "-b"]