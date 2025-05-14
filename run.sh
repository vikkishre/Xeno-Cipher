#!/bin/bash
cd Test
exec gunicorn app:app 