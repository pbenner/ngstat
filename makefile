
SUBDIRS = \
	classification \
	config \
	estimation \
	io \
	statistics/scalarDistribution \
	statistics/scalarEstimator \
	track \
	trackDataTransform \
	tools/ngstat \
	utility

all:

test:
	@for i in $(SUBDIRS); do \
		echo "Testing $$i"; (cd $$i && go test); \
	done
