"""
Main file.
"""

from models.trainer import Trainer
import yaml
import argparse

if __name__ == '__main__':
    trainer = Trainer()
    trainer.train()