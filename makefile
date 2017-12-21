
SUBDIRS = \
	classification \
	config \
	io \
	statistics \
	statistics/generic \
	statistics/matrixDistribution \
	statistics/matrixEstimator \
	statistics/scalarDistribution \
	statistics/scalarEstimator \
	statistics/vectorDistribution \
	statistics/vectorEstimator \
	statistics/matrixClassifier \
	statistics/vectorClassifier \
	statistics/scalarClassifier \
	track \
	trackDataTransform \
	utility \
	estimation \
	tools/ngstat \

all:

test:
	@for i in $(SUBDIRS); do \
		echo "Testing $$i"; (cd $$i && go test); \
	done
