#!/bin/bash
#SBATCH -q short
#SBATCH --nodes=10
#SBATCH --ntasks-per-node=8
#SBATCH -t 48:00:00
#SBATCH -J cspic

julia --project=. driver.jl
