ARG BASE_IMAGE=uniflexai/tinynav@sha256:96f802435261b774da2a72d7c965aa50137195491adc295a345bde0b0c06afa2
FROM ${BASE_IMAGE}

COPY --chmod=0755 docker/entrypoint.sh /usr/local/bin/tinynav-sim

WORKDIR /workspace/tinynav-sim
ENTRYPOINT ["/usr/local/bin/tinynav-sim"]
CMD ["shell"]
