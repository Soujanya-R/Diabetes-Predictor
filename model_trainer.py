import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.metrics import classification_report, confusion_matrix
import xgboost as xgb
from imblearn.over_sampling import SMOTE
import warnings
warnings.filterwarnings('ignore')

class ModelTrainer:
    def __init__(self):
        self.models = {}
        self.model_scores = {}
        self.cv_scores = {}
        self.trained = False
        
        # Initialize models with optimized hyperparameters
        self.initialize_models()
    
    def initialize_models(self):
        """Initialize machine learning models with optimized parameters"""
        self.models = {
            'random_forest': RandomForestClassifier(
                n_estimators=200,
                max_depth=10,
                min_samples_split=5,
                min_samples_leaf=2,
                max_features='sqrt',
                random_state=42,
                class_weight='balanced'
            ),
            'xgboost': xgb.XGBClassifier(
                n_estimators=200,
                max_depth=6,
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
                eval_metric='logloss',
                scale_pos_weight=1.5  # Handle class imbalance
            ),
            'svm': SVC(
                kernel='rbf',
                C=1.0,
                gamma='scale',
                probability=True,  # Enable probability predictions
                random_state=42,
                class_weight='balanced'
            )
        }
    
    def apply_smote(self, X_train, y_train):
        """Apply SMOTE for handling class imbalance"""
        try:
            smote = SMOTE(random_state=42, k_neighbors=3)
            X_resampled, y_resampled = smote.fit_resample(X_train, y_train)
            return X_resampled, y_resampled
        except:
            # If SMOTE fails (e.g., not enough samples), return original data
            return X_train, y_train
    
    def train_models(self, X_train, y_train, cv_folds=5, use_smote=True):
        """Train all models with cross-validation"""
        print("Training models...")
        
        # Apply SMOTE if requested
        if use_smote:
            X_train_balanced, y_train_balanced = self.apply_smote(X_train, y_train)
        else:
            X_train_balanced, y_train_balanced = X_train, y_train
        
        # Cross-validation setup
        cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
        
        for name, model in self.models.items():
            print(f"Training {name}...")
            
            # Perform cross-validation
            cv_scores = cross_val_score(
                model, X_train_balanced, y_train_balanced, 
                cv=cv, scoring='accuracy', n_jobs=-1
            )
            
            self.cv_scores[name] = cv_scores
            
            # Train final model on full training set
            model.fit(X_train_balanced, y_train_balanced)
            
            print(f"{name} - CV Accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
        
        self.trained = True
        print("All models trained successfully!")
    
    def evaluate_models(self, X_test, y_test):
        """Evaluate all trained models on test set"""
        if not self.trained:
            raise ValueError("Models must be trained before evaluation")
        
        print("Evaluating models...")
        
        for name, model in self.models.items():
            # Make predictions
            y_pred = model.predict(X_test)
            y_prob = model.predict_proba(X_test)[:, 1]
            
            # Calculate metrics
            metrics = {
                'accuracy': accuracy_score(y_test, y_pred),
                'precision': precision_score(y_test, y_pred),
                'recall': recall_score(y_test, y_pred),
                'f1': f1_score(y_test, y_pred),
                'roc_auc': roc_auc_score(y_test, y_prob)
            }
            
            self.model_scores[name] = metrics
            
            print(f"\n{name} Results:")
            for metric, score in metrics.items():
                print(f"  {metric}: {score:.4f}")
        
        print("\nModel evaluation completed!")
    
    def get_ensemble_prediction(self, X):
        """Get ensemble prediction using soft voting"""
        if not self.trained:
            raise ValueError("Models must be trained before prediction")
        
        # Get probabilities from all models
        probabilities = []
        predictions = []
        
        for name, model in self.models.items():
            prob = model.predict_proba(X)[:, 1]
            pred = model.predict(X)
            
            probabilities.append(prob)
            predictions.append(pred)
        
        # Ensemble probability (average of all model probabilities)
        ensemble_prob = np.mean(probabilities, axis=0)
        
        # Ensemble prediction (majority voting)
        ensemble_pred = (ensemble_prob >= 0.5).astype(int)
        
        # Individual model results
        individual_results = {
            'probabilities': {name: prob for name, prob in zip(self.models.keys(), probabilities)},
            'predictions': {name: pred for name, pred in zip(self.models.keys(), predictions)}
        }
        
        return ensemble_pred, ensemble_prob, individual_results
    
    def get_feature_importance(self):
        """Get feature importance from tree-based models"""
        if not self.trained:
            raise ValueError("Models must be trained before getting feature importance")
        
        importance_dict = {}
        
        # Random Forest feature importance
        if 'random_forest' in self.models:
            rf_importance = self.models['random_forest'].feature_importances_
            importance_dict['random_forest'] = rf_importance
        
        # XGBoost feature importance
        if 'xgboost' in self.models:
            xgb_importance = self.models['xgboost'].feature_importances_
            importance_dict['xgboost'] = xgb_importance
        
        return importance_dict
    
    def get_model_comparison(self):
        """Get detailed comparison of all models"""
        if not self.model_scores:
            raise ValueError("Models must be evaluated before comparison")
        
        comparison_df = pd.DataFrame(self.model_scores).T
        comparison_df = comparison_df.round(4)
        
        # Add rankings
        for metric in comparison_df.columns:
            comparison_df[f'{metric}_rank'] = comparison_df[metric].rank(ascending=False)
        
        return comparison_df
    
    def get_best_model(self, metric='accuracy'):
        """Get the best performing model based on specified metric"""
        if not self.model_scores:
            raise ValueError("Models must be evaluated before determining best model")
        
        best_model_name = max(self.model_scores.keys(), 
                             key=lambda x: self.model_scores[x][metric])
        
        return best_model_name, self.models[best_model_name]
    
    def predict_with_uncertainty(self, X):
        """Predict with uncertainty quantification"""
        if not self.trained:
            raise ValueError("Models must be trained before prediction")
        
        # Get predictions from all models
        ensemble_pred, ensemble_prob, individual_results = self.get_ensemble_prediction(X)
        
        # Calculate prediction uncertainty
        prob_std = np.std([prob for prob in individual_results['probabilities'].values()], axis=0)
        
        # Confidence based on agreement between models
        pred_agreement = np.mean([pred for pred in individual_results['predictions'].values()], axis=0)
        confidence = np.where(pred_agreement > 0.5, pred_agreement, 1 - pred_agreement)
        
        return {
            'predictions': ensemble_pred,
            'probabilities': ensemble_prob,
            'uncertainty': prob_std,
            'confidence': confidence,
            'individual_results': individual_results
        }
    
    def generate_classification_report(self, X_test, y_test):
        """Generate comprehensive classification report"""
        if not self.trained:
            raise ValueError("Models must be trained before generating report")
        
        reports = {}
        
        for name, model in self.models.items():
            y_pred = model.predict(X_test)
            report = classification_report(y_test, y_pred, output_dict=True)
            reports[name] = report
        
        return reports
    
    def get_confusion_matrices(self, X_test, y_test):
        """Get confusion matrices for all models"""
        if not self.trained:
            raise ValueError("Models must be trained before generating confusion matrices")
        
        confusion_matrices = {}
        
        for name, model in self.models.items():
            y_pred = model.predict(X_test)
            cm = confusion_matrix(y_test, y_pred)
            confusion_matrices[name] = cm
        
        return confusion_matrices
