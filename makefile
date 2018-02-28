
SUBDIRS_TOOLS = \
	tools/ngstat

SUBDIRS = \
	$(SUBDIRS_TOOLS) \
	classification \
	config \
	estimation \
	io \
	statistics/nonparametric \
	track \
	trackDataTransform \
	utility

all:

install:
	@for i in $(SUBDIRS_TOOLS); do \
		echo "Installing $$i"; (cd $$i && go install); \
	done

test:
	@for i in $(SUBDIRS); do \
		echo "Testing $$i"; (cd $$i && go test); \
	done
