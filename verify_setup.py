#!/usr/bin/env python3
"""
Verify setup for Llama 70B evaluation
Checks dependencies, GPU availability, HF access, and data availability
"""

import sys
from pathlib import Path

def check_imports():
    """Check if required packages are installed"""
    print("="*80)
    print("CHECKING PYTHON PACKAGES")
    print("="*80)
    
    required = [
        'torch',
        'transformers',
        'accelerate',
        'sentence_transformers',
        'rank_bm25',
        'datasets',
    ]
    
    optional = [
        'bitsandbytes',  # For 4-bit quantization
    ]
    
    missing = []
    for package in required:
        try:
            __import__(package)
            print(f"✅ {package}")
        except ImportError:
            print(f"❌ {package} - MISSING")
            missing.append(package)
    
    print("\nOptional packages:")
    for package in optional:
        try:
            __import__(package)
            print(f"✅ {package}")
        except ImportError:
            print(f"⚠️  {package} - Not installed (needed for 4-bit quantization)")
    
    if missing:
        print(f"\n❌ Missing required packages: {', '.join(missing)}")
        print("Install with: pip install -r requirements.txt")
        return False
    
    print("\n✅ All required packages installed")
    return True

def check_gpu():
    """Check GPU availability and memory"""
    print("\n" + "="*80)
    print("CHECKING GPU AVAILABILITY")
    print("="*80)
    
    try:
        import torch
        
        if not torch.cuda.is_available():
            print("❌ CUDA not available")
            return False
        
        gpu_count = torch.cuda.device_count()
        print(f"✅ Found {gpu_count} GPU(s)")
        
        total_memory = 0
        for i in range(gpu_count):
            props = torch.cuda.get_device_properties(i)
            memory_gb = props.total_memory / 1e9
            total_memory += memory_gb
            print(f"   GPU {i}: {props.name} - {memory_gb:.1f} GB")
        
        print(f"\nTotal GPU Memory: {total_memory:.1f} GB")
        
        # Check if enough memory for Llama 70B
        if total_memory >= 140:
            print("✅ Sufficient memory for Llama 70B FP16 (~140GB)")
        elif total_memory >= 80:
            print("⚠️  Sufficient memory for Llama 70B 4-bit (~40GB)")
            print("   Recommendation: Use --model llama-3.1-70b-4bit")
        else:
            print("❌ Insufficient memory for Llama 70B")
            print(f"   Need: 140GB (FP16) or 80GB (4-bit)")
            print(f"   Have: {total_memory:.1f}GB")
            return False
        
        return True
        
    except Exception as e:
        print(f"❌ Error checking GPU: {e}")
        return False

def check_huggingface():
    """Check Hugging Face authentication"""
    print("\n" + "="*80)
    print("CHECKING HUGGING FACE ACCESS")
    print("="*80)
    
    try:
        from huggingface_hub import HfApi
        api = HfApi()
        user_info = api.whoami()
        print(f"✅ Logged in as: {user_info['name']}")
        
        # Check if can access Llama 3.1 70B
        try:
            model_id = "meta-llama/Llama-3.1-70B-Instruct"
            model_info = api.model_info(model_id)
            print(f"✅ Access to {model_id} confirmed")
            return True
        except Exception as e:
            print(f"❌ Cannot access {model_id}")
            print(f"   Error: {e}")
            print("\n   Steps to fix:")
            print("   1. Go to https://huggingface.co/meta-llama/Llama-3.1-70B-Instruct")
            print("   2. Accept the license agreement")
            print("   3. Run: huggingface-cli login")
            return False
            
    except Exception as e:
        print(f"❌ Not logged in to Hugging Face")
        print(f"   Error: {e}")
        print("\n   Steps to fix:")
        print("   1. Go to https://huggingface.co/settings/tokens")
        print("   2. Create a token with read access")
        print("   3. Run: huggingface-cli login")
        print("   4. Paste your token")
        return False

def check_data():
    """Check if required data files exist"""
    print("\n" + "="*80)
    print("CHECKING DATA AVAILABILITY")
    print("="*80)
    
    base_path = Path(__file__).parent
    
    required_files = {
        'NFCorpus corpus': 'IPI_generators/ipi_nfcorpus/nfcorpus_ipi_mixed_v2.jsonl',
        'NFCorpus metadata': 'IPI_generators/ipi_nfcorpus/nfcorpus_ipi_metadata_v2.jsonl',
        'NFCorpus queries': 'data/corpus/beir/nfcorpus/queries.jsonl',
        'SciFact corpus': 'IPI_generators/ipi_scifact/scifact_ipi_mixed_v2.jsonl',
        'SciFact metadata': 'IPI_generators/ipi_scifact/scifact_ipi_metadata_v2.jsonl',
        'SciFact queries': 'data/corpus/beir/scifact/queries.jsonl',
    }
    
    all_exist = True
    for name, rel_path in required_files.items():
        full_path = base_path / rel_path
        if full_path.exists():
            size_mb = full_path.stat().st_size / 1e6
            print(f"✅ {name}: {size_mb:.1f} MB")
        else:
            print(f"❌ {name}: NOT FOUND")
            print(f"   Expected at: {full_path}")
            all_exist = False
    
    if not all_exist:
        print("\n❌ Some data files are missing")
        print("   Make sure Phase 1 attack generation is complete")
        return False
    
    print("\n✅ All required data files present")
    return True

def check_scripts():
    """Check if new scripts are present"""
    print("\n" + "="*80)
    print("CHECKING NEW SCRIPTS")
    print("="*80)
    
    base_path = Path(__file__).parent
    
    scripts = {
        'Model configs': 'configs/model_configs.py',
        'Llama 70B evaluator': 'evaluation/run_stages_6_7_llama70b.py',
        'Batch runner': 'evaluation/batch_run_llama70b_all_corpora.py',
        'Requirements': 'requirements.txt',
        'README': 'README.md',
        'Migration guide': 'MIGRATION_LLAMA70B.md',
    }
    
    all_exist = True
    for name, rel_path in scripts.items():
        full_path = base_path / rel_path
        if full_path.exists():
            print(f"✅ {name}")
        else:
            print(f"❌ {name}: NOT FOUND")
            all_exist = False
    
    if not all_exist:
        print("\n❌ Some scripts are missing")
        return False
    
    print("\n✅ All new scripts present")
    return True

def main():
    print("""
╔══════════════════════════════════════════════════════════════════════════════╗
║                                                                              ║
║                   GuardRAG Setup Verification                                ║
║                   Llama 3.1 70B Readiness Check                              ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝
    """)
    
    checks = [
        ('Python Packages', check_imports),
        ('GPU Availability', check_gpu),
        ('Hugging Face Access', check_huggingface),
        ('Data Files', check_data),
        ('Scripts', check_scripts),
    ]
    
    results = {}
    for name, check_func in checks:
        try:
            results[name] = check_func()
        except Exception as e:
            print(f"\n❌ Error during {name} check: {e}")
            results[name] = False
    
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    
    all_passed = True
    for name, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{status} - {name}")
        if not passed:
            all_passed = False
    
    print("\n" + "="*80)
    
    if all_passed:
        print("✅ ALL CHECKS PASSED - Ready to run Llama 70B evaluation!")
        print("\nNext steps:")
        print("1. Read MIGRATION_LLAMA70B.md for detailed instructions")
        print("2. Run quick test:")
        print("   python evaluation/run_stages_6_7_llama70b.py \\")
        print("       --corpus scifact \\")
        print("       --model llama-3.1-70b \\")
        print("       --retriever bm25 \\")
        print("       --sample-size 5")
        return 0
    else:
        print("❌ SOME CHECKS FAILED - Please fix the issues above")
        print("\nRefer to:")
        print("- requirements.txt for package installation")
        print("- MIGRATION_LLAMA70B.md for setup guide")
        print("- README.md for project overview")
        return 1

if __name__ == '__main__':
    sys.exit(main())
