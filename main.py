#!/usr/bin/env python3
"""
GuardRAG Main Entry Point
Provides CLI for running RAG pipeline with defense mechanisms.
"""

import argparse
import logging
from pathlib import Path
from typing import Optional

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def setup_logging(verbose: bool = False) -> None:
    """Configure logging level based on verbosity."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.getLogger().setLevel(level)


def main() -> None:
    """Main entry point for GuardRAG."""
    parser = argparse.ArgumentParser(
        description="GuardRAG: Guarded Retrieval-Augmented Generation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run RAG pipeline with default settings
  python main.py --mode rag --corpus hotpotqa
  
  # Run evaluation on specific dataset
  python main.py --mode eval --dataset fiqa
  
  # Run DPO training
  python main.py --mode dpo --config configs/dpo_config.yaml
        """
    )
    
    parser.add_argument(
        "--mode",
        choices=["rag", "eval", "dpo", "generate", "analyze"],
        default="rag",
        help="Operation mode"
    )
    
    parser.add_argument(
        "--corpus",
        choices=["hotpotqa", "fiqa", "nfcorpus", "scifact", "msmarco", "natural_questions"],
        default="hotpotqa",
        help="Corpus to use"
    )
    
    parser.add_argument(
        "--dataset",
        type=str,
        help="Path to dataset file"
    )
    
    parser.add_argument(
        "--config",
        type=str,
        help="Path to configuration file"
    )
    
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("./output"),
        help="Output directory for results"
    )
    
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable verbose logging"
    )
    
    args = parser.parse_args()
    setup_logging(args.verbose)
    
    # Ensure output directory exists
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Starting GuardRAG in {args.mode} mode")
    logger.info(f"Corpus: {args.corpus}")
    
    try:
        if args.mode == "rag":
            logger.info("Running RAG pipeline...")
            # Import and run RAG pipeline
            from rag_pipeline.pipeline import RAGPipeline
            pipeline = RAGPipeline(corpus=args.corpus, output_dir=args.output_dir)
            pipeline.run()
            
        elif args.mode == "eval":
            logger.info("Running evaluation...")
            # Import and run evaluation
            from evaluation.metrics_calculator import EvaluationRunner
            evaluator = EvaluationRunner(corpus=args.corpus, output_dir=args.output_dir)
            evaluator.run()
            
        elif args.mode == "dpo":
            logger.info("Running DPO training...")
            # Import and run DPO training
            from DPO.dpo.train_dpo import train_dpo_model
            train_dpo_model(config_path=args.config, output_dir=args.output_dir)
            
        elif args.mode == "generate":
            logger.info("Generating data...")
            # Import and run generation
            from IPI_generators.batch_generate_all_corpora import generate_all_corpora
            generate_all_corpora(output_dir=args.output_dir)
            
        elif args.mode == "analyze":
            logger.info("Running analysis...")
            # Import and run analysis
            from IPI_generators.analyze_ipi_statistics import analyze_statistics
            analyze_statistics(corpus=args.corpus, output_dir=args.output_dir)
            
        logger.info("Operation completed successfully!")
        
    except Exception as e:
        logger.error(f"Error during execution: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    main()
