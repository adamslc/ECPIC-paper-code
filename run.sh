#!/bin/bash
#SBATCH -q short
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=8
#SBATCH -t 24:00:00
#SBATCH -J mcpic1

julia --project=. driver.jl
