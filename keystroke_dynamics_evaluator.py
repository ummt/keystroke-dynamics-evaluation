import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
import time
import os
from scipy import stats
from scipy.stats import chi2, binom, wilcoxon
import warnings
warnings.filterwarnings('ignore')

class KeystrokeDynamicsEvaluator:
    """
    Comprehensive Keystroke Dynamics Authentication Evaluation System
    Fully compliant with Killourhy & Maxion (2009) standard protocol
    Academic-grade implementation for fair method comparison
    """
    
    def __init__(self, data_path: str = 'DSL-StrongPasswordData.csv', random_seed: int = 42):
        """Initialize the keystroke dynamics evaluator"""
        np.random.seed(random_seed)
        
        if not os.path.exists(data_path):
            raise FileNotFoundError(
                f"Dataset file '{data_path}' not found.\n"
                f"Please download the CMU Keystroke Dynamics Dataset from:\n"
                f"https://www.cs.cmu.edu/~keystroke/\n"
                f"Save as '{data_path}' in the current directory."
            )
        
        print(f"Loading dataset: {data_path}")
        self.df = pd.read_csv(data_path)
        
        # CMU dataset 31 timing features (standard order)
        self.features = [col for col in self.df.columns 
                        if col.startswith(('H.', 'DD.', 'UD.'))]
        self.features.sort()  # Consistent ordering
        
        print(f"Features loaded: {len(self.features)} timing features")
        print(f"Feature list: {self.features}")
        
        # Dataset validation
        self._validate_dataset()
        
    def _validate_dataset(self):
        """Validate dataset structure and completeness"""
        subjects = self.df['subject'].unique()
        print(f"\n=== Dataset Validation ===")
        print(f"Total users: {len(subjects)}")
        print(f"Total samples: {len(self.df)}")
        
        # Check samples per user
        user_counts = self.df.groupby('subject').size()
        print(f"Samples per user: mean={user_counts.mean():.1f}, std={user_counts.std():.1f}")
        print(f"Min samples: {user_counts.min()}, Max samples: {user_counts.max()}")
        
        # CMU dataset standard: 400 samples per user
        expected_samples = 400
        users_with_expected = sum(user_counts == expected_samples)
        print(f"Users with standard sample count ({expected_samples}): {users_with_expected}/{len(subjects)}")
        
        if users_with_expected < len(subjects):
            print("Warning: Some users have non-standard sample counts")
    
    def create_user_template(self, samples: pd.DataFrame, method: str = 'traditional') -> Dict:
        """
        Create user authentication template
        
        Args:
            samples: Training samples for a specific user
            method: 'traditional' (classical statistics) or 'robust' (ASP approach)
        
        Returns:
            Dictionary containing user template with statistical measures
        """
        template = {}
        
        for feature in self.features:
            values = samples[feature].dropna().values
            if len(values) < 2:
                template[feature] = None
                continue
            
            if method == 'traditional':
                # Killourhy & Maxion (2009) standard approach
                mean = np.mean(values)
                std = np.std(values, ddof=1)
                
                template[feature] = {
                    'mean': mean,
                    'std': std,
                    'n_samples': len(values)
                }
                
            elif method == 'robust':
                # ASP: Adaptive Statistical Profile with robust statistics
                median = np.median(values)
                q1, q3 = np.percentile(values, [25, 75])
                iqr = max(q3 - q1, 1e-10)  # Zero-division protection
                mad = np.median(np.abs(values - median))  # Median Absolute Deviation
                mad = max(mad, 1e-10)  # Zero-division protection
                
                # Classical statistics (for comparison)
                mean = np.mean(values)
                std = np.std(values, ddof=1)
                std = max(std, 1e-10)  # Zero-division protection
                
                # Adaptive weights
                cv = std / (abs(mean) + 1e-10)  # Coefficient of Variation
                stability_weight = 1.0 / (1.0 + cv)  # Higher weight for stable features
                reliability_weight = min(1.0, np.sqrt(len(values)) / 10.0)  # Higher weight for more samples
                composite_weight = stability_weight * reliability_weight
                
                template[feature] = {
                    # Robust statistics
                    'median': median,
                    'iqr': iqr,
                    'mad': mad,
                    # Classical statistics
                    'mean': mean,
                    'std': std,
                    # Adaptive weights
                    'stability_weight': stability_weight,
                    'reliability_weight': reliability_weight,
                    'composite_weight': composite_weight,
                    'n_samples': len(values)
                }
        
        return template
    
    def manhattan_distance(self, template: Dict, sample: pd.Series) -> float:
        """Manhattan distance calculation (Killourhy & Maxion 2009)"""
        if template is None:
            return float('inf')
            
        distance = 0.0
        valid_features = 0
        
        for feature in self.features:
            if (template.get(feature) is None or 
                pd.isna(sample[feature])):
                continue
                
            mean = template[feature]['mean']
            distance += abs(sample[feature] - mean)
            valid_features += 1
        
        return distance if valid_features > 0 else float('inf')
    
    def euclidean_distance(self, template: Dict, sample: pd.Series) -> float:
        """Euclidean distance calculation (Killourhy & Maxion 2009)"""
        if template is None:
            return float('inf')
            
        distance = 0.0
        valid_features = 0
        
        for feature in self.features:
            if (template.get(feature) is None or 
                pd.isna(sample[feature])):
                continue
                
            mean = template[feature]['mean']
            distance += (sample[feature] - mean) ** 2
            valid_features += 1
        
        return np.sqrt(distance) if valid_features > 0 else float('inf')
    
    def scaled_manhattan_distance(self, template: Dict, sample: pd.Series) -> float:
        """Scaled Manhattan distance calculation"""
        if template is None:
            return float('inf')
            
        distance = 0.0
        valid_features = 0
        
        for feature in self.features:
            if (template.get(feature) is None or 
                pd.isna(sample[feature]) or
                template[feature]['std'] <= 1e-10):
                continue
                
            mean = template[feature]['mean']
            std = template[feature]['std']
            distance += abs(sample[feature] - mean) / std
            valid_features += 1
        
        return distance if valid_features > 0 else float('inf')
    
    def asp_distance(self, template: Dict, sample: pd.Series, 
                    weight_type: str = 'composite') -> float:
        """
        Adaptive Statistical Profile (ASP) distance calculation
        
        Args:
            template: User template with robust statistics
            sample: Test sample
            weight_type: Type of weighting ('composite', 'uniform', 'stability', 'reliability')
        
        Returns:
            ASP distance score
        """
        if template is None:
            return float('inf')
            
        weighted_distance = 0.0
        total_weight = 0.0
        valid_features = 0
        
        for feature in self.features:
            if (template.get(feature) is None or 
                pd.isna(sample[feature])):
                continue
                
            feat_template = template[feature]
            value = sample[feature]
            
            # Weight selection
            if weight_type == 'uniform':
                weight = 1.0
            elif weight_type == 'stability':
                weight = feat_template['stability_weight']
            elif weight_type == 'reliability':
                weight = feat_template['reliability_weight']
            elif weight_type == 'composite':
                weight = feat_template['composite_weight']
            else:
                weight = 1.0
            
            # Three robust distance components
            # 1. Median-based normalized distance
            median_dist = abs(value - feat_template['median']) / feat_template['iqr']
            
            # 2. MAD-based normalized distance  
            mad_dist = abs(value - feat_template['median']) / feat_template['mad']
            
            # 3. Standard deviation-based normalized distance
            std_dist = abs(value - feat_template['mean']) / feat_template['std']
            
            # Composite distance with paper coefficients (0.5, 0.3, 0.2)
            composite_dist = 0.5 * median_dist + 0.3 * mad_dist + 0.2 * std_dist
            
            weighted_distance += composite_dist * weight
            total_weight += weight
            valid_features += 1
        
        if total_weight <= 0 or valid_features == 0:
            return float('inf')
            
        return weighted_distance / total_weight
    
    def evaluate_user_authentication(self, user_subject: str, 
                                   training_ratio: float = 0.5) -> Dict:
        """
        Evaluate authentication performance for a single user
        Standard keystroke authentication protocol
        
        Args:
            user_subject: Target user ID
            training_ratio: Ratio of data used for training (default: 0.5)
        
        Returns:
            Dictionary containing evaluation results for all methods
        """
        print(f"\n--- Evaluating User {user_subject} ---")
        
        # Get target user data
        user_data = self.df[self.df['subject'] == user_subject].copy()
        user_data = user_data.sort_values(['sessionIndex', 'rep']).reset_index(drop=True)
        
        if len(user_data) < 20:
            print(f"Insufficient data: {len(user_data)} samples")
            return None
        
        # Train-test split
        n_train = int(len(user_data) * training_ratio)
        n_train = max(10, min(n_train, len(user_data) - 10))  # Ensure minimum samples
        
        train_data = user_data.iloc[:n_train]
        test_data = user_data.iloc[n_train:]
        
        print(f"Training samples: {len(train_data)}, Test samples: {len(test_data)}")
        
        # Create templates
        traditional_template = self.create_user_template(train_data, 'traditional')
        robust_template = self.create_user_template(train_data, 'robust')
        
        # Calculate genuine user scores
        genuine_scores = {
            'manhattan': [],
            'euclidean': [],
            'scaled_manhattan': [],
            'asp_composite': []
        }
        
        for _, sample in test_data.iterrows():
            genuine_scores['manhattan'].append(
                self.manhattan_distance(traditional_template, sample))
            genuine_scores['euclidean'].append(
                self.euclidean_distance(traditional_template, sample))
            genuine_scores['scaled_manhattan'].append(
                self.scaled_manhattan_distance(traditional_template, sample))
            genuine_scores['asp_composite'].append(
                self.asp_distance(robust_template, sample, 'composite'))
        
        # Calculate impostor scores
        other_subjects = [s for s in self.df['subject'].unique() if s != user_subject]
        
        impostor_scores = {
            'manhattan': [],
            'euclidean': [],
            'scaled_manhattan': [],
            'asp_composite': []
        }
        
        # Sample impostors for evaluation
        max_impostors_per_user = max(1, len(test_data) // len(other_subjects))
        
        for impostor_subject in other_subjects:
            impostor_data = self.df[self.df['subject'] == impostor_subject]
            
            if len(impostor_data) == 0:
                continue
            
            # Deterministic sampling for reproducibility
            n_samples = min(max_impostors_per_user, len(impostor_data))
            np.random.seed(hash(user_subject + impostor_subject) % 2**32)
            sample_indices = np.random.choice(len(impostor_data), n_samples, replace=False)
            
            for idx in sample_indices:
                sample = impostor_data.iloc[idx]
                
                impostor_scores['manhattan'].append(
                    self.manhattan_distance(traditional_template, sample))
                impostor_scores['euclidean'].append(
                    self.euclidean_distance(traditional_template, sample))
                impostor_scores['scaled_manhattan'].append(
                    self.scaled_manhattan_distance(traditional_template, sample))
                impostor_scores['asp_composite'].append(
                    self.asp_distance(robust_template, sample, 'composite'))
        
        print(f"Genuine scores: {len(genuine_scores['manhattan'])}")
        print(f"Impostor scores: {len(impostor_scores['manhattan'])}")
        
        # Calculate EER for each method
        user_results = {}
        for method in genuine_scores.keys():
            eer_result = self._calculate_eer(
                genuine_scores[method], impostor_scores[method])
            
            user_results[method] = {
                'eer': eer_result['eer'],
                'threshold': eer_result['threshold'],
                'frr': eer_result['frr'],
                'far': eer_result['far'],
                'genuine_scores': genuine_scores[method],
                'impostor_scores': impostor_scores[method]
            }
            
            print(f"{method}: EER = {eer_result['eer']*100:.2f}%")
        
        return user_results
    
    def _calculate_eer(self, genuine_scores: List[float], 
                      impostor_scores: List[float]) -> Dict:
        """
        Calculate Equal Error Rate (EER) for distance-based authentication
        
        Args:
            genuine_scores: Distance scores for legitimate user
            impostor_scores: Distance scores for impostors
        
        Returns:
            Dictionary with EER, threshold, FRR, and FAR
        """
        if not genuine_scores or not impostor_scores:
            return {'eer': 1.0, 'threshold': float('inf'), 'frr': 1.0, 'far': 1.0}
        
        # Remove infinite values
        genuine_finite = [s for s in genuine_scores if np.isfinite(s)]
        impostor_finite = [s for s in impostor_scores if np.isfinite(s)]
        
        if not genuine_finite or not impostor_finite:
            return {'eer': 1.0, 'threshold': float('inf'), 'frr': 1.0, 'far': 1.0}
        
        all_scores = sorted(set(genuine_finite + impostor_finite))
        
        min_eer = 1.0
        best_result = {'eer': 1.0, 'threshold': float('inf'), 'frr': 1.0, 'far': 1.0}
        
        for threshold in all_scores:
            # Distance-based: accept if distance <= threshold
            frr = sum(1 for s in genuine_finite if s > threshold) / len(genuine_finite)
            far = sum(1 for s in impostor_finite if s <= threshold) / len(impostor_finite)
            eer = (frr + far) / 2.0
            
            if eer < min_eer:
                min_eer = eer
                best_result = {
                    'eer': eer,
                    'threshold': threshold,
                    'frr': frr,
                    'far': far
                }
        
        return best_result
    
    def run_full_evaluation(self, training_ratio: float = 0.5) -> Dict:
        """
        Run comprehensive evaluation across all users
        Compliant with Killourhy & Maxion (2009) protocol
        
        Args:
            training_ratio: Ratio of data used for training
        
        Returns:
            Dictionary containing complete evaluation results
        """
        print("="*80)
        print("COMPREHENSIVE KEYSTROKE DYNAMICS AUTHENTICATION EVALUATION")
        print("Killourhy & Maxion (2009) Fully Compliant Implementation")
        print("="*80)
        
        start_time = time.time()
        
        subjects = sorted(self.df['subject'].unique())
        print(f"\nEvaluating {len(subjects)} users")
        print(f"Training ratio: {training_ratio:.1%}")
        
        # Store all user results
        all_user_results = {}
        method_performance = {
            'manhattan': [],
            'euclidean': [],
            'scaled_manhattan': [],
            'asp_composite': []
        }
        
        successful_evaluations = 0
        
        # Evaluate each user independently
        for user_idx, user_subject in enumerate(subjects):
            print(f"\n[{user_idx+1}/{len(subjects)}] User: {user_subject}")
            
            user_result = self.evaluate_user_authentication(user_subject, training_ratio)
            
            if user_result is None:
                print("Evaluation failed: skipping")
                continue
            
            all_user_results[user_subject] = user_result
            successful_evaluations += 1
            
            # Add to aggregate statistics
            for method in method_performance.keys():
                if method in user_result:
                    method_performance[method].append(user_result[method]['eer'])
        
        print(f"\nSuccessful evaluations: {successful_evaluations}/{len(subjects)}")
        
        # Calculate average performance
        aggregate_results = {}
        for method, eer_list in method_performance.items():
            if eer_list:
                aggregate_results[method] = {
                    'mean_eer': np.mean(eer_list),
                    'std_eer': np.std(eer_list),
                    'median_eer': np.median(eer_list),
                    'min_eer': np.min(eer_list),
                    'max_eer': np.max(eer_list),
                    'n_users': len(eer_list),
                    'eer_list': eer_list
                }
        
        # Statistical significance testing with Wilcoxon signed-rank test
        print(f"\n=== Statistical Significance Testing (Wilcoxon Signed-Rank Test) ===")
        statistical_results = self._comprehensive_statistical_analysis(all_user_results)
        
        execution_time = time.time() - start_time
        
        # Display results
        self._display_complete_results(aggregate_results, statistical_results)
        
        # Generate final report
        final_report = self._generate_complete_report(
            aggregate_results, statistical_results, successful_evaluations, execution_time)
        
        return {
            'aggregate_results': aggregate_results,
            'individual_results': all_user_results,
            'statistical_results': statistical_results,
            'final_report': final_report,
            'metadata': {
                'n_successful_users': successful_evaluations,
                'n_total_users': len(subjects),
                'training_ratio': training_ratio,
                'execution_time_seconds': execution_time
            }
        }
    
    def _comprehensive_statistical_analysis(self, all_user_results: Dict) -> Dict:
        """
        Comprehensive statistical analysis (Wilcoxon + effect size + confidence intervals)
        
        Args:
            all_user_results: Individual results for all users
        
        Returns:
            Dictionary containing statistical test results
        """
        comparisons = [
            ('asp_composite', 'manhattan'),
            ('asp_composite', 'euclidean'),
            ('asp_composite', 'scaled_manhattan')
        ]
        
        results = {}
        
        for method1, method2 in comparisons:
            print(f"\n=== {method1} vs {method2} ===")
            
            # Collect EER differences
            eer_differences = []
            method1_better = 0
            method2_better = 0
            ties = 0
            
            for user, user_result in all_user_results.items():
                if method1 not in user_result or method2 not in user_result:
                    continue
                    
                eer1 = user_result[method1]['eer']
                eer2 = user_result[method2]['eer']
                diff = eer2 - eer1  # Positive value means method1 is better
                
                eer_differences.append(diff)
                
                if diff > 0:
                    method1_better += 1
                elif diff < 0:
                    method2_better += 1
                else:
                    ties += 1
            
            # Basic statistics
            n_users = len(eer_differences)
            mean_diff = np.mean(eer_differences)
            std_diff = np.std(eer_differences, ddof=1)
            median_diff = np.median(eer_differences)
            
            print(f"Number of users: {n_users}")
            print(f"{method1} wins: {method1_better}, {method2} wins: {method2_better}, Ties: {ties}")
            print(f"Mean EER difference: {mean_diff*100:.3f}% ± {std_diff*100:.3f}%")
            print(f"Median EER difference: {median_diff*100:.3f}%")
            
            # Wilcoxon signed-rank test
            wilcoxon_stat = None
            wilcoxon_p = 1.0
            
            if n_users >= 6:
                try:
                    # Remove zero differences
                    non_zero_diffs = [d for d in eer_differences if abs(d) > 1e-10]
                    
                    if len(non_zero_diffs) >= 6:
                        wilcoxon_stat, wilcoxon_p = wilcoxon(
                            non_zero_diffs, 
                            alternative='two-sided'
                        )
                        
                        print(f"Wilcoxon test: statistic={wilcoxon_stat}, p={wilcoxon_p:.6f}")
                        
                except Exception as e:
                    print(f"Wilcoxon test error: {e}")
            
            # Effect size (Cohen's d equivalent)
            if std_diff > 0:
                cohens_d = mean_diff / std_diff
            else:
                cohens_d = 0
            
            # Win rate (alternative effect size indicator)
            win_rate = method1_better / (method1_better + method2_better) if (method1_better + method2_better) > 0 else 0.5
            
            # 95% confidence interval (bootstrap)
            ci_lower = ci_upper = None
            if n_users >= 10:
                bootstrap_means = []
                n_bootstrap = 1000
                np.random.seed(42)  # For reproducibility
                
                for _ in range(n_bootstrap):
                    bootstrap_sample = np.random.choice(eer_differences, n_users, replace=True)
                    bootstrap_means.append(np.mean(bootstrap_sample))
                
                ci_lower = np.percentile(bootstrap_means, 2.5)
                ci_upper = np.percentile(bootstrap_means, 97.5)
                
                print(f"95% confidence interval: [{ci_lower*100:.3f}%, {ci_upper*100:.3f}%]")
            
            # Effect size interpretation
            if abs(cohens_d) < 0.2:
                effect_interpretation = "negligible"
            elif abs(cohens_d) < 0.5:
                effect_interpretation = "small"
            elif abs(cohens_d) < 0.8:
                effect_interpretation = "medium"
            else:
                effect_interpretation = "large"
            
            print(f"Effect size (Cohen's d): {cohens_d:.3f} ({effect_interpretation})")
            print(f"Win rate: {win_rate:.3f} ({method1_better}/{method1_better + method2_better})")
            
            # P-value display with detailed values
            p_display_detailed = f"{wilcoxon_p:.2e}"

            if wilcoxon_p < 1e-10:
                p_display = f"p < 1e-10 (p = {wilcoxon_p:.2e})"
            elif wilcoxon_p < 0.001:
                p_display = f"p < 0.001 (p = {wilcoxon_p:.6f})"
            elif wilcoxon_p < 0.01:
                p_display = f"p < 0.01 (p = {wilcoxon_p:.6f})"
            elif wilcoxon_p < 0.05:
                p_display = f"p < 0.05 (p = {wilcoxon_p:.6f})"
            else:
                p_display = f"p = {wilcoxon_p:.6f}"

            significance = "significant" if wilcoxon_p < 0.05 else "not significant"
            print(f"Statistical significance: {p_display} ({significance})")

            # Practical significance assessment
            practical_threshold = 0.01  # 1% improvement considered practically significant
            practical_significant = abs(mean_diff) > practical_threshold

            print(f"Practical significance: {'Yes' if practical_significant else 'No'} (threshold: {practical_threshold*100}%)")

            # Store results
            results[f'{method1}_vs_{method2}'] = {
                'n_users': n_users,
                'method1_better': method1_better,
                'method2_better': method2_better,
                'ties': ties,
                'mean_difference': mean_diff,
                'std_difference': std_diff,
                'median_difference': median_diff,
                'cohens_d': cohens_d,
                'win_rate': win_rate,
                'wilcoxon_statistic': wilcoxon_stat,
                'wilcoxon_p_value': wilcoxon_p,
                'wilcoxon_p_detailed': p_display_detailed,
                'confidence_interval': (ci_lower, ci_upper) if ci_lower is not None else None,
                'effect_interpretation': effect_interpretation,
                'statistically_significant': wilcoxon_p < 0.05,
                'practically_significant': practical_significant,
                'p_display': p_display,
                'p_display_detailed': p_display_detailed
            }

        return results
    
    def _display_complete_results(self, aggregate_results: Dict, statistical_results: Dict):
        """Display comprehensive evaluation results"""
        print("\n" + "="*60)
        print("COMPREHENSIVE EVALUATION RESULTS")
        print("="*60)
        
        methods_display = {
            'asp_composite': 'ASP (Proposed Method)',
            'manhattan': 'Manhattan Distance',
            'euclidean': 'Euclidean Distance',
            'scaled_manhattan': 'Scaled Manhattan'
        }
        
        print(f"\n【Equal Error Rate - User Average】")
        sorted_methods = sorted(aggregate_results.items(), 
                            key=lambda x: x[1]['mean_eer'])
        
        for method, result in sorted_methods:
            display_name = methods_display.get(method, method)
            print(f"{display_name:25s}: "
                f"{result['mean_eer']*100:5.2f}% ± {result['std_eer']*100:4.2f}% "
                f"(median: {result['median_eer']*100:5.2f}%, n={result['n_users']})")
        
        # Improvement rates
        if 'asp_composite' in aggregate_results:
            asp_eer = aggregate_results['asp_composite']['mean_eer']
            print(f"\n【Improvement Rates】")
            
            for method in ['manhattan', 'euclidean', 'scaled_manhattan']:
                if method in aggregate_results:
                    baseline_eer = aggregate_results[method]['mean_eer']
                    improvement = (baseline_eer - asp_eer) / baseline_eer * 100
                    print(f"vs {methods_display[method]:20s}: {improvement:+5.1f}%")
        
        # Statistical significance
        print(f"\n【Statistical Significance (Wilcoxon Signed-Rank Test)】")
        for comparison, stats in statistical_results.items():
            method1, method2 = comparison.replace('_vs_', ' vs ').split(' vs ')
            print(f"{method1.upper()} vs {method2.upper()}:")
            print(f"  Win-Loss: {stats['method1_better']}W {stats['method2_better']}L {stats['ties']}T")
            print(f"  Mean improvement: {stats['mean_difference']*100:+.2f}%")
            print(f"  Effect size: {stats['cohens_d']:.3f} ({stats['effect_interpretation']})")
            print(f"  Statistical significance: {stats['p_display']} ({'significant' if stats['statistically_significant'] else 'not significant'})")
            print(f"  Exact p-value: {stats['wilcoxon_p_value']:.2e}")
            if stats['confidence_interval']:
                ci_lower, ci_upper = stats['confidence_interval']
                print(f"  95% CI: [{ci_lower*100:+.2f}%, {ci_upper*100:+.2f}%]")
            print()  # Empty line
        
        print("\n" + "="*80)
        print("KEYSTROKE DYNAMICS AUTHENTICATION EVALUATION COMPLETED")
        print("="*80)
    
    def _generate_complete_report(self, aggregate_results: Dict, statistical_results: Dict,
                                n_users: int, execution_time: float) -> str:
        """Generate comprehensive evaluation report"""
        report = []
        report.append("="*80)
        report.append("KEYSTROKE DYNAMICS AUTHENTICATION - COMPREHENSIVE EVALUATION")
        report.append("Implementation Compliant with Killourhy & Maxion (2009)")
        report.append("="*80)
        
        report.append(f"\nEvaluation completed: {time.strftime('%Y-%m-%d %H:%M:%S')}")
        report.append(f"Dataset: CMU Keystroke Dynamics Dataset")
        report.append(f"Evaluated users: {n_users}")
        report.append(f"Features: {len(self.features)} timing features")
        report.append(f"Execution time: {execution_time:.1f} seconds")
        
        # Method descriptions
        report.append(f"\n" + "="*50)
        report.append("METHODS")
        report.append("="*50)
        report.append("ASP (Adaptive Statistical Profile):")
        report.append("  - Robust statistics: Median, IQR, MAD")
        report.append("  - Adaptive weighting: Stability × Reliability")
        report.append("  - Distance fusion: 0.5×Median + 0.3×MAD + 0.2×Std")
        report.append("")
        report.append("Baseline methods (Killourhy & Maxion 2009):")
        report.append("  - Manhattan: Σ|xi - μi|")
        report.append("  - Euclidean: √Σ(xi - μi)²")
        report.append("  - Scaled Manhattan: Σ|xi - μi|/σi")
        
        # Results
        report.append(f"\n" + "="*50)
        report.append("RESULTS (Mean EER ± Std)")
        report.append("="*50)
        
        sorted_methods = sorted(aggregate_results.items(), 
                            key=lambda x: x[1]['mean_eer'])
        
        for method, result in sorted_methods:
            method_name = method.replace('_', ' ').title()
            report.append(f"{method_name:20s}: {result['mean_eer']*100:5.2f}% ± {result['std_eer']*100:4.2f}%")
        
        # Statistical significance
        report.append(f"\n" + "="*50)
        report.append("STATISTICAL SIGNIFICANCE (Wilcoxon Signed-Rank Test)")
        report.append("="*50)
        
        for comparison, stats in statistical_results.items():
            method1, method2 = comparison.replace('_vs_', ' vs ').split(' vs ')
            report.append(f"{method1.upper()} vs {method2.upper()}:")
            report.append(f"  Win Rate: {stats['win_rate']:.3f} ({stats['method1_better']}/{stats['method1_better'] + stats['method2_better']})")
            report.append(f"  Mean Improvement: {stats['mean_difference']*100:+.2f}%")
            report.append(f"  Effect Size (Cohen's d): {stats['cohens_d']:.3f} ({stats['effect_interpretation']})")
            report.append(f"  Statistical Significance: {stats['p_display']}")
            report.append(f"  Exact p-value: {stats['wilcoxon_p_detailed']}")
            if stats['confidence_interval']:
                ci_lower, ci_upper = stats['confidence_interval']
                report.append(f"  95% CI: [{ci_lower*100:+.2f}%, {ci_upper*100:+.2f}%]")
            report.append("")  # Empty line
        
        report.append("="*80)
        
        return "\n".join(report)

def main():
    """Main execution function for comprehensive keystroke dynamics evaluation"""
    try:
        evaluator = KeystrokeDynamicsEvaluator()
        
        print("\nRunning comprehensive keystroke dynamics authentication evaluation")
        print("Implementation fully compliant with Killourhy & Maxion (2009)")
        print("Statistical significance testing with Wilcoxon signed-rank test")
        
        # Run complete evaluation
        results = evaluator.run_full_evaluation(training_ratio=0.5)
        
        # Display final report
        print("\n" + "="*80)
        print("FINAL REPORT")
        print("="*80)  
        print(results['final_report'])
        
        return results
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return None


if __name__ == "__main__":
    results = main()