mkdir -p services_logs
if [ -f cog_setup/cog.env ]
then
  export $(cat cog_setup/cog.env | xargs)
else
  exit 1
fi
nohup /opt/miniconda3/bin/conda run --live-stream -n WavJourney python services.py > services_logs/service.out 2>&1 &
