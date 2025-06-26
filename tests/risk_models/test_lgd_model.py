import unittest
from unittest.mock import MagicMock, patch
import pandas as pd
import numpy as np
from pathlib import Path
import tempfile
import joblib
import datetime

from src.risk_models.lgd_model import LGDModel
from src.data_management.knowledge_base import KnowledgeBaseService
from src.data_management.ontology import (
    LoanAgreement, CollateralType, Currency # Removed SeniorityOfDebt
)
# Assuming ModelRegistry might be used, though not explicitly in LGDModel train's current state
from src.mlops.model_registry import ModelRegistry
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder


class TestLGDModel(unittest.TestCase):

    def setUp(self):
        self.temp_model_dir = tempfile.TemporaryDirectory()
        self.test_model_path = Path(self.temp_model_dir.name) / "test_lgd_model.joblib"
        self.lgd_model = LGDModel(model_path=self.test_model_path)

        self.mock_kb_service = MagicMock(spec=KnowledgeBaseService)

        # Sample data for mocking KB - focusing on defaulted loans for LGD
        self.loan1 = LoanAgreement(
            loan_id="L1_default", company_id="C1", loan_amount=100000, currency=Currency.USD, # Changed CurrencyCode to Currency
            origination_date=datetime.date(2022, 1, 1), maturity_date=datetime.date(2024, 1, 1),
            interest_rate_percentage=5.0, collateral_type=CollateralType.REAL_ESTATE,
            seniority_of_debt="Senior", default_status=True,
            collateral_value_usd=120000, recovery_rate_percentage=0.7, # Actual historical data
            economic_condition_indicator=0.6
        )
        self.loan2 = LoanAgreement(
            loan_id="L2_default", company_id="C2", loan_amount=50000, currency=Currency.USD, # Changed CurrencyCode to Currency
            origination_date=datetime.date(2023, 1, 1), maturity_date=datetime.date(2023, 6, 1),
            interest_rate_percentage=8.0, collateral_type=CollateralType.INVENTORY,
            seniority_of_debt="Junior", default_status=True,
            collateral_value_usd=30000, recovery_rate_percentage=0.3,
            economic_condition_indicator=0.4
        )
        self.loan3_nodefault = LoanAgreement( # Non-defaulted, should be ignored by _prepare_features
            loan_id="L3_nodefault", company_id="C3", loan_amount=200000, currency=Currency.GBP, # Changed CurrencyCode to Currency
            origination_date=datetime.date(2023, 3, 1), maturity_date=datetime.date(2026, 3, 1),
            interest_rate_percentage=3.5, collateral_type=CollateralType.EQUIPMENT,
            seniority_of_debt="Senior", default_status=False,
            economic_condition_indicator=0.5
        )
        self.loan4_default_none_collateral = LoanAgreement(
            loan_id="L4_default_none", company_id="C4", loan_amount=75000, currency=Currency.USD, # Changed CurrencyCode to Currency
            origination_date=datetime.date(2022, 5, 1), maturity_date=datetime.date(2023, 5, 1),
            interest_rate_percentage=6.0, collateral_type=CollateralType.NONE, # Test None collateral
            seniority_of_debt="Unsecured", default_status=True,
            economic_condition_indicator=0.3
        )


        self.mock_kb_service.get_all_loans.return_value = [self.loan1, self.loan2, self.loan3_nodefault, self.loan4_default_none_collateral]

    def tearDown(self):
        self.temp_model_dir.cleanup()

    def test_prepare_features_and_target_structure_and_values(self):
        """Test _prepare_features_and_target method for LGD."""
        # Mock np.random.normal to make recovery rate predictable
        with patch('src.risk_models.lgd_model.np.random.normal', return_value=0.01): # Fixed noise
            features_df = self.lgd_model._prepare_features_and_target(self.mock_kb_service)

        self.assertIsInstance(features_df, pd.DataFrame)
        # Only defaulted loans should be included (L1, L2, L4)
        self.assertEqual(len(features_df), 3)

        expected_cols = [
            'loan_id', 'loan_amount_usd', 'collateral_type',
            'seniority_of_debt', 'economic_condition_indicator', 'recovery_rate'
        ]
        for col in expected_cols:
            self.assertIn(col, features_df.columns)

        # Check values for loan L1
        l1_features = features_df[features_df['loan_id'] == 'L1_default'].iloc[0]
        self.assertEqual(l1_features['collateral_type'], CollateralType.REAL_ESTATE.value)
        self.assertEqual(l1_features['seniority_of_debt'], "Senior") # Changed to string "Senior"
        self.assertEqual(l1_features['economic_condition_indicator'], 0.6)

        # Calculate expected recovery rate for L1 (Real Estate, Senior, Econ 0.6, Noise 0.01)
        # base_recovery = 0.7 (Real Estate)
        # seniority_adj = +0.10 (Senior)
        # econ_adj = (0.6 - 0.5) * 0.2 = 0.02
        # noise = 0.01
        # expected_rec = 0.7 + 0.10 + 0.02 + 0.01 = 0.83. Clipped to [0.05, 0.95] -> 0.83
        self.assertAlmostEqual(l1_features['recovery_rate'], 0.83, places=4)


        # Check L4 (None collateral)
        l4_features = features_df[features_df['loan_id'] == 'L4_default_none'].iloc[0]
        self.assertEqual(l4_features['collateral_type'], CollateralType.NONE.value) # Should be 'None' string
         # base_recovery = 0.1 (Default)
        # seniority_adj for UNSECURED (not explicitly handled, treated as 'Unknown' or other, so no adjustment from list)
        # Let's trace: base_recovery = 0.1. Seniority 'Unsecured' -> no specific adj.
        # econ_indicator = 0.3. econ_adj = (0.3-0.5)*0.2 = -0.04
        # noise = 0.01
        # expected_rec = 0.1 + 0 - 0.04 + 0.01 = 0.07. Clipped to [0.05, 0.95] -> 0.07
        self.assertAlmostEqual(l4_features['recovery_rate'], 0.07, places=4)


    def test_prepare_features_no_defaulted_loans(self):
        """Test _prepare_features when no loans are defaulted."""
        self.loan1.default_status = False
        self.loan2.default_status = False
        self.loan4_default_none_collateral.default_status = False
        self.mock_kb_service.get_all_loans.return_value = [self.loan1, self.loan2, self.loan3_nodefault, self.loan4_default_none_collateral]
        features_df = self.lgd_model._prepare_features_and_target(self.mock_kb_service)
        self.assertTrue(features_df.empty)

    @patch('src.risk_models.lgd_model.ModelRegistry')
    @patch('src.risk_models.lgd_model.joblib.dump')
    def test_train_successful(self, mock_joblib_dump, MockModelRegistry):
        """Test successful LGD model training."""
        mock_registry_instance = MockModelRegistry.return_value
        # Ensure enough defaulted loans for train/test split
        # self.loan1, self.loan2, self.loan4 are defaulted. Total 3.
        # test_size = 0.2 means 0.6 sample for test, so 1 sample. Train = 2 samples.
        # This should be enough for GBR not to complain.

        metrics = self.lgd_model.train(self.mock_kb_service, test_size=0.2) # Ensure test_size is small

        self.assertIn('train_mse', metrics)
        self.assertIn('test_mse', metrics)
        self.assertNotIn('error', metrics)
        self.assertIsNotNone(self.lgd_model.model)
        mock_joblib_dump.assert_called_once_with(self.lgd_model.model, self.test_model_path)

        mock_registry_instance.register_model.assert_called_once()
        args, kwargs = mock_registry_instance.register_model.call_args
        self.assertEqual(kwargs['model_name'], "LGDModel")


    def test_train_insufficient_data(self):
        """Test LGD training with insufficient data (e.g., after filtering)."""
        self.mock_kb_service.get_all_loans.return_value = [self.loan3_nodefault] # Only one non-defaulted loan
        metrics = self.lgd_model.train(self.mock_kb_service)
        self.assertIn('error', metrics)
        self.assertIn("Insufficient data", metrics['error'])
        self.assertIsNone(self.lgd_model.model)

    def test_predict_lgd(self):
        """Test LGD prediction after training a model."""
        # Train a model (simplified way for testing predict)
        with patch('src.risk_models.lgd_model.ModelRegistry'), patch('src.risk_models.lgd_model.joblib.dump'):
            self.lgd_model.train(self.mock_kb_service)
        self.assertIsNotNone(self.lgd_model.model)

        sample_features = {
            'collateral_type': 'Real Estate',
            'loan_amount_usd': 150000,
            'seniority_of_debt': 'Senior',
            'economic_condition_indicator': 0.65
        }
        predicted_lgd = self.lgd_model.predict_lgd(sample_features)
        self.assertIsInstance(predicted_lgd, float)
        self.assertTrue(0.0 <= predicted_lgd <= 1.0) # LGD must be between 0 and 1

        # Test with missing new features (should take defaults)
        sample_features_missing = {
            'collateral_type': 'Equipment',
            'loan_amount_usd': 75000
        }
        predicted_lgd_missing = self.lgd_model.predict_lgd(sample_features_missing)
        self.assertIsInstance(predicted_lgd_missing, float)
        self.assertTrue(0.0 <= predicted_lgd_missing <= 1.0)


    def test_predict_lgd_no_model(self):
        """Test LGD prediction when model is not trained/loaded."""
        self.assertIsNone(self.lgd_model.model)
        sample_features = {'collateral_type': 'None', 'loan_amount_usd': 100}
        # Default LGD is 0.75 if model not loaded
        self.assertEqual(self.lgd_model.predict_lgd(sample_features), 0.75)

    def test_save_and_load_model(self):
        """Test saving and loading the LGD model."""
        # Train a dummy model for saving
        df = pd.DataFrame({
            'loan_amount_usd': [100, 200, 150],
            'collateral_type': ['None', 'Real Estate', 'Inventory'],
            'seniority_of_debt': ['Senior', 'Junior', 'Senior'],
            'economic_condition_indicator': [0.5, 0.3, 0.7],
            'recovery_rate': [0.1, 0.6, 0.4] # Target
        })
        X = df.drop('recovery_rate', axis=1)
        y = df['recovery_rate']

        # Temporarily set numerical and categorical features for this dummy model
        original_num_feats = self.lgd_model.numerical_features
        original_cat_feats = self.lgd_model.categorical_features

        current_num_feats = ['loan_amount_usd', 'economic_condition_indicator']
        current_cat_feats = ['collateral_type', 'seniority_of_debt']
        self.lgd_model.numerical_features = current_num_feats
        self.lgd_model.categorical_features = current_cat_feats

        # Rebuild the preprocessor part of the base_model with current feature lists
        # Get the original regressor
        regressor_step = self.lgd_model.base_model.steps[1] # ('regressor', GradientBoostingRegressor(...))

        new_preprocessor = ColumnTransformer(
            transformers=[
                ('num', Pipeline([('imputer', SimpleImputer(strategy='median'))]), current_num_feats),
                ('cat', Pipeline([('imputer', SimpleImputer(strategy='most_frequent')),
                                  ('onehot', OneHotEncoder(handle_unknown='ignore'))]), current_cat_feats)
            ]
        )
        self.lgd_model.base_model = Pipeline(steps=[('preprocessor', new_preprocessor), regressor_step])

        self.lgd_model.model = self.lgd_model.base_model.fit(X,y)
        self.lgd_model.save_model()
        self.assertTrue(self.test_model_path.exists())

        new_lgd_model = LGDModel(model_path=self.test_model_path)
        load_success = new_lgd_model.load_model()
        self.assertTrue(load_success)
        self.assertIsNotNone(new_lgd_model.model)

        # Restore original features
        self.lgd_model.numerical_features = original_num_feats
        self.lgd_model.categorical_features = original_cat_feats


    def test_load_model_not_exists(self):
        """Test loading LGD model that does not exist (no registry fallback)."""
        non_existent_path = Path(self.temp_model_dir.name) / "ghost_lgd_model.joblib"
        new_lgd_model = LGDModel(model_path=non_existent_path)
        with patch('src.risk_models.lgd_model.ModelRegistry') as MockReg:
            MockReg.return_value.get_production_model_path.return_value = None
            load_success = new_lgd_model.load_model()
        self.assertFalse(load_success)
        self.assertIsNone(new_lgd_model.model)

    @patch('src.risk_models.lgd_model.joblib.load')
    @patch('src.risk_models.lgd_model.ModelRegistry')
    def test_load_model_fallback_to_registry(self, MockModelRegistry, mock_joblib_load):
        """Test LGD model loading falls back to registry."""
        non_existent_path = Path(self.temp_model_dir.name) / "ghost_lgd_model.joblib"
        registry_model_path_str = str(Path(self.temp_model_dir.name) / "registry_lgd_model.joblib")

        # Create a simple, real scikit-learn pipeline to dump for the registry model
        # Need to import these for this dummy pipeline
        from sklearn.pipeline import Pipeline
        from sklearn.impute import SimpleImputer
        dummy_skl_pipeline = Pipeline([('imputer', SimpleImputer(strategy='mean'))])
        # Create some dummy data to fit it so it's a valid "trained" model
        dummy_X = pd.DataFrame({'feature': [1,2,3,np.nan]})
        dummy_skl_pipeline.fit(dummy_X)


        with open(registry_model_path_str, 'wb') as f:
            joblib.dump(dummy_skl_pipeline, f)

        mock_registry_instance = MockModelRegistry.return_value
        mock_registry_instance.get_production_model_path.return_value = registry_model_path_str
        mock_joblib_load.return_value = dummy_skl_pipeline # joblib.load will return this object

        new_lgd_model = LGDModel(model_path=non_existent_path)
        load_success = new_lgd_model.load_model()

        self.assertTrue(load_success)
        self.assertIsNotNone(new_lgd_model.model)
        mock_registry_instance.get_production_model_path.assert_called_once_with("LGDModel")
        mock_joblib_load.assert_called_once_with(Path(registry_model_path_str))
        self.assertEqual(new_lgd_model.model_path, Path(registry_model_path_str))


if __name__ == '__main__':
    unittest.main()
