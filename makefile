
SUBDIRS = \
	classification \
	config \
	io \
	statistics/scalarDistribution \
	statistics/scalarEstimator \
	track \
	trackDataTransform \
	utility \
	estimation \
	tools/ngstat

all:

test:
	@for i in $(SUBDIRS); do \
		echo "Testing $$i"; (cd $$i && go test); \
	done
