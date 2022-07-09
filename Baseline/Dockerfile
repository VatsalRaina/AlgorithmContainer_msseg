FROM pytorch/pytorch



RUN groupadd -r algorithm && useradd -m --no-log-init -r -g algorithm algorithm

RUN mkdir -p /opt/algorithm /input /output \
    && chown algorithm:algorithm /opt/algorithm /input /output

USER algorithm

WORKDIR /opt/algorithm

ENV PATH="/home/algorithm/.local/bin:${PATH}"

RUN python -m pip install --user -U pip



COPY --chown=algorithm:algorithm requirements.txt /opt/algorithm/
RUN python -m pip install --user -rrequirements.txt

COPY --chown=algorithm:algorithm process.py /opt/algorithm/
COPY --chown=algorithm:algorithm model1.pth /opt/algorithm/model1.pth
COPY --chown=algorithm:algorithm model2.pth /opt/algorithm/model2.pth
COPY --chown=algorithm:algorithm model3.pth /opt/algorithm/model3.pth

ENTRYPOINT python -m process $0 $@
