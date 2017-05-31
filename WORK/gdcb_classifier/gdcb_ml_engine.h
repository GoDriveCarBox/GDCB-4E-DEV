//
// GoDriveCarBox Machine Learning Engine
//
// Created: 		2017-01-30
// Last modified: 	2017-05-30
//
// @copyright: 4E SOFTWARE SRL
// 

#pragma once

#include "stdafx.h"
#include "stdio.h"
#include <string>
#include <iostream>
#include <fstream>
#include <vector>
#include <set>

#include <Eigen/Dense>
#include <Eigen/Core>
#include <Eigen/SVD>

#include <sys/stat.h>

#include <chrono>

#include <algorithm>
#include <random>

using namespace std;
using namespace std::chrono;


using Eigen::MatrixXd;
using Eigen::VectorXd;

using namespace Eigen;
using namespace std;


struct TrainCrossSplits
{
	MatrixXd X_train;
	MatrixXd X_cross;
	vector <string> Labels;
	VectorXd y_train;
	VectorXd y_cross;
};



class GenericEngine
{
private:
	long LoadedDataNrFields;
	long LoadedDataNrRows;
	long TrainTestSplitPos;

	std::default_random_engine random_engine;

protected:
	milliseconds start_time;
	milliseconds end_time;

	bool bBiasAdded; // variable that stores bias information for pre-loaded data

	string CLF_NAME;
	long NR_FEATS;
	long NR_CLASSES;

public:

	bool VERBOSE_ENGINE;

	MatrixXd *X_loaded;
	VectorXd *y_loaded;

	MatrixXd *X_train;
	VectorXd *y_train;

	MatrixXd *X_cross;
	VectorXd *y_cross;

	MatrixXd *LoadedData;
	vector <string> LoadedDataHeader;
	vector <string> LabelsVector;


	GenericEngine()
	{
		CLF_NAME = "Generic Engine";

		// obtain a time-based seed:
		unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
		random_engine = default_random_engine(seed);

		VERBOSE_ENGINE = true;

		bBiasAdded = false;

		X_train = NULL;
		y_train = NULL;
		X_loaded = NULL;
		y_loaded = NULL;
		X_cross = NULL;
		y_cross = NULL;
		LoadedData = NULL;
	}

	bool file_exists(const std::string& name);


	void debug_info(string str_message)
	{
		if (VERBOSE_ENGINE)
			printf("\n[DEBUG] %s", str_message.c_str());
	}

	void debug_info(string msg, MatrixXd mat)
	{
		std::stringstream ss;
		ss << mat;
		string str_matrix = ss.str();
		string msgp = "\n[DEBUG] " + msg + "\n";
		printf(msgp.c_str());
		std::cout << str_matrix << std::endl;
	}


	void debug_info(MatrixXd mat)
	{
		std::stringstream ss;
		ss << mat;
		string str_matrix = ss.str();
		printf("\n[DEBUG] Matrix:\n");
		std::cout << str_matrix << std::endl;
	}

	void debug_info(VectorXd vec, bool bHorizontal)
	{
		std::stringstream ss;
		if (bHorizontal)
			ss << vec.transpose();
		else
			ss << vec;
		string str_vector = ss.str();
		printf("\n[DEBUG] Vector:\n");
		std::cout << str_vector << std::endl;
	}

	void debug_info(string msg, VectorXd vec, bool bHorizontal)
	{
		std::stringstream ss;
		if (bHorizontal)
			ss << vec.transpose();
		else
			ss << vec;
		string str_vector = ss.str();
		string msgp = "\n[DEBUG] " + msg + "\n";
		printf(msgp.c_str());
		std::cout << str_vector << std::endl;
	}
	void debug_info()
	{

		printf("\n[DEBUG] [PRESS ENTER]");
		int c = getc(stdin);

	}


	MatrixXd ShuffleMatrixRows(MatrixXd DataMatrix);


	int FindLabelId(vector <string> labels, string value);

	void BeginTimer();
	long EndTimer();

	vector <string> ToLabels(VectorXd y);

	TrainCrossSplits LoadCSV(const string& inputfile, const bool bShuffle = false, const bool bAddBias = false);

};

class GenericLinearEngine : public GenericEngine
{
protected:

	long nr_batches; // how many batches have been processed (epochs, online trainings, etc)
	VectorXd *SingleClassTheta;
	MatrixXd *Theta;
	VectorXd *J_values;

	void add_cost(double J);


private:

	void init();

public:



	GenericLinearEngine()
	{
		CLF_NAME = "VIRTUAL Generic Linear Engine";
		init();

	}
	~GenericLinearEngine()
	{
		debug_info("Deleting object [" + CLF_NAME + "]");

		if (LoadedData != NULL)
			delete LoadedData;
		if (X_loaded != NULL)
			delete X_loaded;
		if (y_loaded != NULL)
			delete y_loaded;
		if (X_train != NULL)
			delete X_train;
		if (y_train != NULL)
			delete y_train;
		if (X_cross != NULL)
			delete X_cross;
		if (y_cross != NULL)
			delete y_cross;
		if (Theta != NULL)
			delete Theta;
		if (SingleClassTheta != NULL)
			delete SingleClassTheta;
		if (J_values != NULL)
			delete J_values;
	}

	VectorXd PredictSingleClass(MatrixXd X);

	virtual MatrixXd Predict(MatrixXd X);

	vector <string> PredictLabels(MatrixXd X);

	vector <string> PredictLabelsUsingYHat(MatrixXd y_hat);

	string GetName();
	MatrixXd& GetTheta();

	float NRMSE(VectorXd y_hat, VectorXd y);
	float RMSE(VectorXd y_hat, VectorXd y);
	float CrossEvaluationSingleClass(bool bClass);
	float TrainEvaluationSingleClass(bool bClass);

	float CrossEvaluation(bool bClass);
	float TrainEvaluation(bool bClass);
};

class NormalRegressor : public GenericLinearEngine
{
protected:
	
	int t;

public:
	NormalRegressor()
	{
		NR_FEATS = 0;
		NR_CLASSES = 0;
		CLF_NAME = "Batch Normal Regressor";
	}

	void Train(MatrixXd X, MatrixXd y);
	void Train();

};

class OnlineClassifier :  public GenericLinearEngine
{
protected:

	// temp variables

	MatrixXd LastYHat;
	MatrixXd LastGrad;
	MatrixXd LastXObs;
	MatrixXd LastYOHM;
	MatrixXd LastYERR;

	double LearningRate;

	MatrixXd softmax(MatrixXd z);
	double cross_entropy(MatrixXd yOHM, MatrixXd y_hat);


public:
	OnlineClassifier(int nr_features, int nr_classes, vector <string> &labels, double alpha_learning_rate)
	{
		CLF_NAME = "Online Linear Classifier";
		NR_FEATS = nr_features;
		NR_CLASSES = nr_classes;
		LabelsVector = labels;
		LearningRate = alpha_learning_rate;

		Theta = new MatrixXd(NR_FEATS+1, NR_CLASSES); // add 1 row for biases

		Theta->fill(0);

	}

	void SimulateOnlineTrain();

	void OnlineTrain(MatrixXd xi, VectorXd yi);

	double CostFunction();

	MatrixXd Predict(MatrixXd X);




};



//
// BEGIN Generic Engine Class definitions - basic ancestor helper class
//

inline bool GenericEngine::file_exists(const std::string & name)
{
	if (FILE *file = fopen(name.c_str(), "r")) {
		fclose(file);
		return true;
	}
	else {
		return false;
	}
}
inline MatrixXd GenericEngine::ShuffleMatrixRows(MatrixXd DataMatrix)
{
	long size = DataMatrix.rows();
	PermutationMatrix<Dynamic, Dynamic> perm(size);
	perm.setIdentity();


	std::shuffle(perm.indices().data(), 
				 perm.indices().data() + perm.indices().size(), 
				 this->random_engine);

	MatrixXd A_perm = perm * DataMatrix; // permute rows
	return(A_perm);
}

inline int GenericEngine::FindLabelId(vector<string> labels, string value)
{
	int pos = find(labels.begin(), labels.end(), value) - labels.begin();

	if (pos >= labels.size()) {
		//old_name_ not found
		pos = -1;
	}

	return(pos);
}
void GenericEngine::BeginTimer()
{
	milliseconds ms = duration_cast< milliseconds >(
		system_clock::now().time_since_epoch()
		);
	start_time = ms;
}

inline long GenericEngine::EndTimer()
{
	milliseconds ms = duration_cast< milliseconds >(
		system_clock::now().time_since_epoch()
		);
	end_time = ms;
	return (end_time - start_time).count();
}

inline vector<string> GenericEngine::ToLabels(VectorXd y)
{
	vector <string> labels;
	for (long i = 0;i < y.size();i++)
	{
		string s = LabelsVector[y(i)];
		labels.push_back(s);
	}
	return(labels);
}

inline TrainCrossSplits GenericEngine::LoadCSV(const string & inputfile, const bool bShuffle, const bool bAddBias)
{
	int nr_rows = 0;
	int nr_cols = 0;
	string fname = inputfile;
	TrainCrossSplits rec_results;

	if (!file_exists(inputfile))
		throw std::invalid_argument("Received invalid file in LoadCSV: " + fname);

	ifstream infile(fname, std::ifstream::in);


	if (!infile.good())
		throw std::invalid_argument("Received invalid file in LoadCSV: " + fname);

	debug_info("Loading " + fname + " dataset...");
	vector< vector<string> > result;
	while (!infile.eof())
	{
		//go through every line
		string line;

		getline(infile, line);

		vector <string> record;
		nr_cols = 0;

		std::size_t prev = 0, pos;
		while ((pos = line.find_first_of(",;", prev)) != std::string::npos)
		{
			if (pos > prev)
			{
				record.push_back(line.substr(prev, pos - prev));
				nr_cols++;
			}

			prev = pos + 1;
		}
		if (prev < line.length())
		{
			record.push_back(line.substr(prev, std::string::npos));
			nr_cols++;
		}


		if (nr_cols > 0)
		{
			result.push_back(record);
			nr_rows++;
		}
	}


	//
	// now load whole data, X and y matrices 
	// assume last column of loaded data is the results / labels
	//

	LoadedDataNrFields = result[0].size();
	LoadedDataNrRows = nr_rows - 1; // rows minus field names row

	debug_info("Loaded " + std::to_string(LoadedDataNrRows) + " X " + std::to_string(LoadedDataNrFields) + " dataset");

	LoadedData = new MatrixXd(LoadedDataNrRows, LoadedDataNrFields);
	y_loaded = new VectorXd(LoadedDataNrRows);
	X_loaded = new MatrixXd(LoadedDataNrRows, LoadedDataNrFields - 1);

	std::set <string> LabelsSet;


	long i, j;
	for (j = 0;j < LoadedDataNrFields;j++)
		LoadedDataHeader.push_back((string)result[0][j]);

	//
	// assume dataset is curated and ONLY last column contains text labels
	//

	vector <string> loaded_labels;

	for (i = 0;i < LoadedDataNrRows;i++)
		for (j = 0;j < LoadedDataNrFields;j++)
		{
			double fcell = 0;
			string scell = result[i + 1][j];
			try
			{
				if (j != ((LoadedDataNrFields - 1)))
					fcell = ::atof(scell.c_str());
			}
			catch (...)
			{

			}
			(*LoadedData)(i, j) = fcell;
			if (j == (LoadedDataNrFields - 1))
			{
				LabelsSet.insert(scell);
				loaded_labels.push_back(scell);
			}

		}

	LabelsVector.assign(LabelsSet.begin(), LabelsSet.end());

	for (int label_idx = 0;label_idx < loaded_labels.size();label_idx++)
	{
		string c_label = loaded_labels[label_idx];
		int iLabel = FindLabelId(LabelsVector, c_label);
		(*LoadedData)(label_idx, LoadedDataNrFields - 1) = iLabel;

	}


	if (bShuffle)
	{
		MatrixXd ttt = ShuffleMatrixRows(*LoadedData);
		*LoadedData = ttt;

	}

	float test_size = 0.2;
	int test_rows = LoadedDataNrRows * test_size;
	int train_rows = LoadedDataNrRows - test_rows;
	TrainTestSplitPos = train_rows;

	*X_loaded = LoadedData->leftCols(LoadedDataNrFields - 1);
	*y_loaded = LoadedData->rightCols(1);

	NR_FEATS = X_loaded->cols();
	NR_CLASSES = LabelsVector.size();


	if (bAddBias)
	{
		// now add bias
		VectorXd bias(LoadedDataNrRows);
		bias.fill(1);
		MatrixXd *TempX = new MatrixXd(LoadedDataNrRows, LoadedDataNrFields - 1 + 1); // bias size 
		*TempX << bias, *X_loaded;
		bBiasAdded = true;
		delete X_loaded;
		X_loaded = TempX;
		// done adding bias
	}

	X_train = new MatrixXd(X_loaded->topRows(train_rows));
	X_cross = new MatrixXd(X_loaded->bottomRows(test_rows));

	y_train = new VectorXd(y_loaded->head(train_rows));
	y_cross = new VectorXd(y_loaded->tail(test_rows));

	rec_results.X_cross = *X_cross;
	rec_results.X_train = *X_train;
	rec_results.y_cross = *y_cross;
	rec_results.y_train = *y_train;
	rec_results.Labels = LabelsVector;

	return(rec_results);

}

//
// END Generig Engine Class definitions
//

//
// BEGIN Generic Linear Engine (Virtual class)
//

inline float GenericLinearEngine::NRMSE(VectorXd y_hat, VectorXd y)
{
	float maxmin = y.maxCoeff()-y.minCoeff();
	return(RMSE(y_hat, y) / maxmin);
}

inline float GenericLinearEngine::RMSE(VectorXd y_hat, VectorXd y)
{
	long nr_obs = y.size();
	VectorXd errors = (y-y_hat);
	if (VERBOSE_ENGINE)
	{
		debug_info("Errors (last 3):");
		debug_info(errors.tail(3));
	}
	double sqNorm = errors.squaredNorm();
	return(sqrt(sqNorm / nr_obs));
}

inline void GenericLinearEngine::add_cost(double J)
{
	if (nr_batches == 0)
	{
		// first use :)
		J_values = new VectorXd(1);
		(*J_values)(nr_batches) = J;
	}
	else
	{
		J_values->conservativeResize(nr_batches + 1);
		(*J_values)(nr_batches) = J;
	}
	nr_batches++;
}

inline void GenericLinearEngine::init()
{

	debug_info("Generating object [" + CLF_NAME + "]");

	nr_batches = 0;
	Theta = NULL;
	SingleClassTheta = NULL;
	J_values = NULL;


}

double myexp(double val)
{
	return(exp(val));
}

MatrixXd& GenericLinearEngine::GetTheta()
{
	return *Theta;
}
string GenericLinearEngine::GetName()
{
	return(CLF_NAME);
}







double myround(double f)
{
	return(round(f));
}

inline VectorXd GenericLinearEngine::PredictSingleClass(MatrixXd X)
{
	VectorXd *pred = new VectorXd(X.rows());

	*pred = X * (*SingleClassTheta);

	return(*pred);
}

inline MatrixXd GenericLinearEngine::Predict(MatrixXd X)
{
	MatrixXd preds = X * (*Theta);
	return(preds);
}

inline vector<string> GenericLinearEngine::PredictLabels(MatrixXd X)
{
	MatrixXd y_hat = Predict(X);
	vector <string> PredictedLabels;
	for (long i = 0;i < y_hat.rows();i++)
	{
		int y_hat_idx;
		y_hat.row(i).maxCoeff(&y_hat_idx);
		PredictedLabels.push_back(LabelsVector[y_hat_idx]);
	}
	return(PredictedLabels);
}

inline vector<string> GenericLinearEngine::PredictLabelsUsingYHat(MatrixXd y_hat)
{
	vector <string> PredictedLabels;
	for (long i = 0;i < y_hat.rows();i++)
	{
		int y_hat_idx;
		y_hat.row(i).maxCoeff(&y_hat_idx);
		PredictedLabels.push_back(LabelsVector[y_hat_idx]);
	}
	return(PredictedLabels);
}


inline float GenericLinearEngine::TrainEvaluationSingleClass(bool bClass)
{
	double dResult = 0.0f;
	VectorXd y = *y_train;
	MatrixXd X = *X_train;
	long nr_train = y.size();
	if (SingleClassTheta == NULL && Theta == NULL)
		return (dResult);

	VectorXd y_hat = PredictSingleClass(X);
	long nr_obs = y_hat.size();

	if (VERBOSE_ENGINE)
	{
		debug_info("Train Y_Hat vs. Y_train (last 3)");
		MatrixXd result(nr_train, 2);
		result << y_hat, y;
		debug_info(result.bottomRows(3));
	}


	if (bClass)
	{
		VectorXd y_hat_Rounded = y_hat.unaryExpr(ptr_fun(myround));
		long positives = 0;
		for (long i = 0;i < nr_obs;i++)
		{
			if (y_hat_Rounded(i) == (y)(i))
				positives++;
		}
		dResult = (double)positives / nr_obs;
	}
	else
	{
		dResult = NRMSE(y_hat, y);
	}

	return (dResult);
}

inline float GenericLinearEngine::CrossEvaluation(bool bClass)
{
	double dResult = 0.0f;
	VectorXd y = *y_cross;
	MatrixXd X;
	if (!bBiasAdded)
		X = *X_cross;
	else
		X = X_cross->rightCols(NR_FEATS);

	long nr_cross = y.size();
	if (Theta == NULL)
		return (dResult);

	MatrixXd y_hat = Predict(X);

	long nr_obs = X.rows();

	if (VERBOSE_ENGINE)
	{
		MatrixXd result(nr_cross, y_hat.cols() + 1);
		result << y_hat, y;
		debug_info("Cross Y_Hat vs. Y_cross (last 5):",result.bottomRows(5));
	}


	if (bClass)
	{
		vector <string> preds = PredictLabelsUsingYHat(y_hat);
		long positives = 0;
		for (long i = 0;i < nr_obs;i++)
		{
			string predicted = preds[i];
			string label = LabelsVector[(int)y(i)];
			if (predicted == label)
				positives++;
		}
		dResult = (double)positives / nr_obs;
	}
	else
	{
		dResult = -1;
	}

	return (dResult);

}

inline float GenericLinearEngine::TrainEvaluation(bool bClass)
{
	double dResult = 0.0f;
	VectorXd y = *y_train;
	MatrixXd X;
	if (!bBiasAdded)
		X = *X_train;
	else
		X = X_train->rightCols(NR_FEATS);

	long nr_cross = y.size();
	if (Theta == NULL)
		return (dResult);

	MatrixXd y_hat = Predict(X);

	long nr_obs = X.rows();

	if (VERBOSE_ENGINE)
	{
		MatrixXd result(nr_cross, y_hat.cols() + 1);
		result << y_hat, y;
		debug_info("Train Y_Hat vs. Y_train (last 5):", result.bottomRows(5));
	}


	if (bClass)
	{
		vector <string> preds = PredictLabelsUsingYHat(y_hat);
		long positives = 0;
		for (long i = 0;i < nr_obs;i++)
		{
			string predicted = preds[i];
			string label = LabelsVector[(int)y(i)];
			if (predicted == label)
				positives++;
		}
		dResult = (double)positives / nr_obs;
	}
	else
	{
		dResult = -1;
	}

	return (dResult);

}

inline float GenericLinearEngine::CrossEvaluationSingleClass(bool bClass)
{
	double dResult = 0.0f;
	VectorXd y = *y_cross;
	MatrixXd X = *X_cross;
	long nr_cross = y.size();
	if (SingleClassTheta == NULL && Theta == NULL)
		return (dResult);
	
	VectorXd y_hat = PredictSingleClass(X);
	long nr_obs = y_hat.size();

	if (VERBOSE_ENGINE)
	{
		debug_info("Cross Y_Hat vs. Y_cross (last 3)");
		MatrixXd result(nr_cross, 2);
		result << y_hat, y;
		debug_info(result.bottomRows(3));
	}


	if (bClass)
	{
		VectorXd y_hat_Rounded = y_hat.unaryExpr(ptr_fun(myround));
		long positives = 0;
		for (long i = 0;i < nr_obs;i++)
		{
			if (y_hat_Rounded(i) == y(i))
				positives++;
		}
		dResult = (double) positives / nr_obs;
	}
	else
	{
		dResult = NRMSE(y_hat, y);
	}

	return (dResult);
}
//
// END Generic Linear Engine virtual class
//



// 
// BEGIN Normal Regressor class definitions
//
void NormalRegressor::Train(MatrixXd X, MatrixXd y)
{
	X_train = new MatrixXd(X);
	y_train = new VectorXd(y);
	Train();
}

template<typename _Matrix_Type_>
_Matrix_Type_ pseudoInverse(const _Matrix_Type_ &a, double epsilon = std::numeric_limits<double>::epsilon())
{
	Eigen::JacobiSVD< _Matrix_Type_ > svd(a, Eigen::ComputeThinU | Eigen::ComputeThinV);
	double tolerance = epsilon * std::max(a.cols(), a.rows()) *svd.singularValues().array().abs()(0);
	return svd.matrixV() *  (svd.singularValues().array().abs() > tolerance).select(svd.singularValues().array().inverse(), 0).matrix().asDiagonal() * svd.matrixU().adjoint();
}

void NormalRegressor::Train()
{
	debug_info("Training: " + CLF_NAME);
	MatrixXd X = *X_train;
	VectorXd y = *y_train;
	MatrixXd xTx = X.transpose() * X;
	MatrixXd xT = X.transpose();

	VectorXd TempTheta1(X.cols());
	VectorXd TempTheta2(X.cols());
	long duration1;
	long duration2;

	if (VERBOSE_ENGINE)
	{
		// 1st solving with pseudo-inverse
		high_resolution_clock::time_point t1 = high_resolution_clock::now();
		MatrixXd xTxInv = pseudoInverse(xTx);
		TempTheta1 = xTxInv * xT * y;
		high_resolution_clock::time_point t2 = high_resolution_clock::now();
		duration1 = duration_cast<microseconds>(t2 - t1).count();



		// now second method
		high_resolution_clock::time_point t3 = high_resolution_clock::now();
		TempTheta2 = xTx.ldlt().solve(xT * y);
		high_resolution_clock::time_point t4 = high_resolution_clock::now();
		duration2 = duration_cast<microseconds>(t4 - t3).count();

		//SingleClassTheta = new VectorXd(TempTheta1);
		SingleClassTheta = new VectorXd(TempTheta2);


	}
	else
	{
		// now second method
		TempTheta2 = xTx.ldlt().solve(xT * y);
		SingleClassTheta = new VectorXd(TempTheta2);
	}


	if (VERBOSE_ENGINE)
	{
		debug_info("X data features size = " + to_string(X_loaded->cols()));

		debug_info("Theta PInv = " + to_string(duration1) + " microsec");
		debug_info("Theta ldlt = " + to_string(duration2) + " microsec");

		debug_info("T1(pinv) T2(ldlt):");
		MatrixXd comp(TempTheta1.size(), 2);
		comp << TempTheta1, TempTheta2;
		debug_info(comp);
		if (*SingleClassTheta == TempTheta2)
			debug_info("Using Theta2");
		else
			debug_info("Using Theta1");

	}
}
//
// END Normal Regressor class definitions
//



inline void OnlineClassifier::SimulateOnlineTrain()
{
	if (Theta != NULL)
		delete Theta;
	Theta = new MatrixXd(NR_FEATS + 1, NR_CLASSES);
	Theta->fill(0); // reset Theta

	long TEST_DEBUG = 1000;

	BeginTimer();

	for (long i = 0;i < X_train->rows();i++)
	{
		MatrixXd obs = X_train->row(i);
		VectorXd yi(1);
		yi(0) = (*y_train)(i);


		if (VERBOSE_ENGINE)// && (i == TEST_DEBUG))
		{
			std::stringstream ss;
			for (size_t i = 0; i < yi.size(); ++i)
			{
				if (i != 0)
					ss << ",";
				ss << yi[i];
			}

			debug_info("Training "+to_string(i)+" th example with y = " + ss.str(),obs);
		}

		MatrixXd xi;
		if (bBiasAdded)
			xi = obs.rightCols(NR_FEATS);
		else
			xi = obs;

		OnlineTrain(xi, yi);

		if (VERBOSE_ENGINE)// && (i == TEST_DEBUG))
		{
			//long time_cost = EndTimer();
			//debug_info("Total time = " + to_string(time_cost) + " ms");
			debug_info("y_OHM (1 row): ",LastYOHM.topRows(1));
			debug_info("y_hat (1 row): ",LastYHat.topRows(1));
			debug_info("error (1 row): ",LastYERR.topRows(1));
			debug_info("Gradient (2 rows): ",LastGrad.topRows(2));

			debug_info("J array las val: ",J_values->tail(1), true);
			debug_info("Theta (2 rows): ",Theta->topRows(2));
			//debug_info();
		}
	}
}

//
// BEGIN Online Classifier definitions
//
// yi is index in VectorLabels
void OnlineClassifier::OnlineTrain(MatrixXd xi, VectorXd yi) 
{
	long nr_rows = xi.rows();
	long nr_cols = xi.cols();

	VectorXd bias(nr_rows);
	bias.fill(1);
	MatrixXd TempX(nr_rows, nr_cols + 1);
	TempX << bias, xi;

	long m = nr_rows; // for convenience
	MatrixXd yOHM(nr_rows, NR_CLASSES);
	yOHM.fill(0);
	for (long i = 0;i < nr_rows;i++)
	{
		for (long j = 0;j < NR_CLASSES;j++)
			// now assume LabelsVector is correctly constructed 
			// and yi[i] is index in that vector
			if (yi(i) == j)
				yOHM(i, j) = 1;
	}

	// now we have the one hot matrix lets start working !
	MatrixXd y_hat = Predict(xi);
	double J = (1.0 / m) * cross_entropy(yOHM, y_hat); // MUST add regularization
	add_cost(J);

	MatrixXd error = yOHM - y_hat;

	MatrixXd Grad = (-1.0 / m) * TempX.transpose() * error; // MUST add regularization

	
	*Theta = *Theta - (LearningRate * Grad);

	LastGrad = Grad;
	LastYOHM = yOHM;
	LastYHat = y_hat;
	LastYERR = error;
	LastXObs = xi;

}

inline double OnlineClassifier::CostFunction()
{

	return 0.0;
}
inline MatrixXd OnlineClassifier::Predict(MatrixXd X)
{
	long nr_rows = X.rows();
	long nr_cols = X.cols();
	
	VectorXd bias(nr_rows);
	bias.fill(1);
	MatrixXd TempX(nr_rows, nr_cols +1);
	TempX << bias, X;
	MatrixXd XTheta = TempX * (*Theta);

	MatrixXd SM = softmax(XTheta);

	return(SM);
}



inline MatrixXd OnlineClassifier::softmax(MatrixXd z)
{
	MatrixXd SM(z.rows(), Theta->cols());

	ArrayXXd arr(z);

	// first shift values
	arr = arr - z.maxCoeff();
	arr = arr.exp();

	//cout << z;
	//cout << arr;

	ArrayXd sums = arr.rowwise().sum();

	arr.colwise() /= sums;

	SM = arr.matrix();


	return(SM);

}

double myclip(double val)
{
	double eps = 1e-15;
	if (val < eps)
		return(eps);
	else
		if (val > (1 - eps))
			return(1 - eps);
		else
			return(val);
}
inline double OnlineClassifier::cross_entropy(MatrixXd yOHM, MatrixXd y_hat)
{
	//y_hat = y_hat.unaryExpr(ptr_fun(myclip));

	MatrixXd J_matrix = (yOHM.array() * y_hat.array().log()).matrix();
	double J = -(J_matrix.sum());
	return(J);
}
//
// END Online Classifier definitions
//

