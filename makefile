
SUBDIRS = \
	classification \
	config \
	estimation \
	io \
	statistics/nonparametric \
	track \
	trackDataTransform \
	tools/ngstat \
	utility

all:

test:
	@for i in $(SUBDIRS); do \
		echo "Testing $$i"; (cd $$i && go test); \
	done
