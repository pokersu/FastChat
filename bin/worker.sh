#!/bin/bash

nohup python -m chatbot.worker 2>&1>/dev/null &
