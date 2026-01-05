"""
Working XTTS Fine-tuning Script
Uses the actual TTS Trainer API properly
"""

import os
import sys
import torch
from pathlib import Path
import json

def create_xtts_config(dataset_path, output_path, epochs=10, batch_size=2):
    """
    Create a proper XTTS config file for training
    """
    
    config = {
        # Model
        "model": "xtts",
        "run_name": "xtts_indian_finetune",
        "project_name": "XTTS_Indian_Celebrities",
        
        # Audio config
        "audio": {
            "sample_rate": 22050,
            "output_sample_rate": 24000,
        },
        
        # Training
        "batch_size": batch_size,
        "eval_batch_size": batch_size,
        "num_loader_workers": 2,
        "num_eval_loader_workers": 1,
        "epochs": epochs,
        "print_step": 50,
        "save_step": 1000,
        "save_n_checkpoints": 2,
        "save_checkpoints": True,
        "save_best_after": 5000,
        "mixed_precision": False,
        
        # Paths
        "output_path": str(output_path),
        
        # Dataset
        "datasets": [
            {
                "name": "vox_indian",
                "path": str(dataset_path),
                "meta_file_train": "metadata.csv",
                "language": "hi",
            }
        ],
        
        # Optimization  
        "lr": 5e-6,  # Low learning rate for fine-tuning
        "grad_clip": 5.0,
        
        # Test
        "test_sentences": [
            "नमस्ते, मैं भारतीय आवाज़ मॉडल हूं।",
            "This is a test of the fine-tuned model.",
            "आज का मौसम बहुत अच्छा है।"
        ],
        
        # Use pre-trained
        "use_phonemes": False,
        "text_cleaner": "multilingual_cleaners",
        "enable_eos_bos_chars": False,
        "phoneme_language": "hi",
    }
    
    return config

def finetune_xtts_real(dataset_path, output_path, epochs=10, batch_size=2):
    """
    Real XTTS fine-tuning using TTS trainer
    """
    
    print("🔥 STARTING REAL XTTS FINE-TUNING")
    print("=" * 70)
    
    dataset_path = Path(dataset_path)
    output_path = Path(output_path)
    
    # Verify dataset
    if not dataset_path.exists():
        print(f"❌ Dataset not found: {dataset_path}")
        return False
    
    metadata_file = dataset_path / "metadata.csv"
    if not metadata_file.exists():
        print(f"❌ Metadata not found: {metadata_file}")
        return False
    
    # Load dataset info
    info_file = dataset_path / "dataset_info.json"
    if info_file.exists():
        with open(info_file) as f:
            info = json.load(f)
        print(f"\n📊 Dataset:")
        print(f"   Speakers: {info['num_speakers']}")
        print(f"   Samples: {info['total_samples']}")
    
    # System info
    print(f"\n💻 System:")
    print(f"   CUDA: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"   GPU: {torch.cuda.get_device_name(0)}")
        print(f"   Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    
    # Create output directory
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Create config
    print(f"\n⚙️  Creating training config...")
    config_dict = create_xtts_config(dataset_path, output_path, epochs, batch_size)
    
    # Save config
    config_file = output_path / "config.json"
    with open(config_file, 'w') as f:
        json.dump(config_dict, f, indent=2)
    print(f"   ✅ Config saved: {config_file}")
    
    # Import TTS training
    try:
        from trainer import Trainer, TrainerArgs
        from TTS.tts.configs.xtts_config import XttsConfig
        from TTS.tts.models.xtts import Xtts
        
        print(f"\n📦 Loading XTTS model...")
        
        # Create proper XttsConfig
        config = XttsConfig()
        config.load_json(str(config_file))
        
        # Initialize model
        model = Xtts(config)
        
        # Try to load pretrained weights
        print(f"   Loading pretrained XTTS v2 weights...")
        try:
            from TTS.utils.manage import ModelManager
            manager = ModelManager()
            model_path = Path(manager.output_prefix) / "tts_models--multilingual--multi-dataset--xtts_v2"
            
            checkpoint_file = model_path / "model.pth"
            if checkpoint_file.exists():
                checkpoint = torch.load(checkpoint_file, map_location='cpu')
                model.load_state_dict(checkpoint, strict=False)
                print(f"   ✅ Loaded pretrained weights")
            else:
                print(f"   ⚠️  Pretrained weights not found, training from scratch")
        except Exception as e:
            print(f"   ⚠️  Could not load pretrained: {e}")
        
        # Setup trainer
        print(f"\n🏋️  Setting up trainer...")
        trainer_args = TrainerArgs(
            restore_path=None,
            skip_train_epoch=False,
        )
        
        trainer = Trainer(
            trainer_args,
            config,
            output_path=str(output_path),
            model=model,
        )
        
        # Start training
        print(f"\n🚀 STARTING TRAINING")
        print(f"   This will take 1-3 hours with GPU...")
        print(f"   Epochs: {epochs}")
        print(f"   Batch size: {batch_size}")
        print("=" * 70)
        print()
        
        trainer.fit()
        
        print("\n" + "=" * 70)
        print("✅ TRAINING COMPLETE!")
        print(f"📁 Model saved to: {output_path}")
        
        return True
        
    except ImportError as e:
        print(f"\n❌ Missing dependency: {e}")
        print("\nPlease install:")
        print("   pip install trainer")
        return False
        
    except Exception as e:
        print(f"\n❌ Training error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Real XTTS fine-tuning')
    parser.add_argument('--dataset', type=str, default='xtts_indian_dataset')
    parser.add_argument('--output', type=str, default='xtts_hindi_finetuned')  
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch-size', type=int, default=2)
    
    args = parser.parse_args()
    
    success = finetune_xtts_real(
        args.dataset,
        args.output,
        args.epochs,
        args.batch_size
    )
    
    if success:
        print("\n🎉 Fine-tuning successful!")
        print("Your model is ready to use in app.py")
    else:
        print("\n❌ Fine-tuning failed")
        sys.exit(1)
