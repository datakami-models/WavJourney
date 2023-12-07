mkdir -p services_logs
if [ -f cog/cog.env ]
then
  export $(cat cog/cog.env | xargs)
fi
nohup /opt/miniconda3/bin/conda run --live-stream -n WavJourney python services.py > services_logs/service.out 2>&1 &
