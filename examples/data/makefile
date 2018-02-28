
# variable definitions
# ------------------------------------------------------------------------------

GENOME  = mm10.genome

BINSIZE = 10
MAPQ    = 30

BAMTOBIGWIG         = bamToBigWig
BAMTOBIGWIG_ARGS    = -v --binning-method="mean overlap" --bin-size=$(BINSIZE) --filter-mapq=$(MAPQ) --filter-duplicates
BAMTOBIGWIG_ARGS_PE = $(BAMTOBIGWIG_ARGS)
BAMTOBIGWIG_ARGS_SE = $(BAMTOBIGWIG_ARGS) --estimate-fragment-length --fragment-length-range=150:250 --fragment-length-bin-size=10 --save-fraglen --save-crosscorrelation --save-crosscorrelation-plot

TARGETS_RPM_ATAC = 
TARGETS_RPM_HM   = 
TARGETS_RPM_CTRL = 

TARGETS_RAW_ATAC = 
TARGETS_RAW_HM   = Liver-Day12.5-H3K27ac.raw.bw
TARGETS_RAW_CTRL = Liver-Day12.5-Control.raw.bw

# main targets
# ------------------------------------------------------------------------------

all: atac hm control

atac: $(TARGETS_RPM_ATAC) $(TARGETS_RAW_ATAC)

hm: $(TARGETS_RPM_HM) $(TARGETS_RAW_HM)

control: $(TARGETS_RPM_CTRL) $(TARGETS_RAW_CTRL)

# link bigWigs to bams
# ------------------------------------------------------------------------------

Liver-Day12.5-H3K27ac.raw.bw: .bam/ENCFF524ZFV.bam .bam/ENCFF322QGS.bam
Liver-Day12.5-Control.raw.bw: .bam/ENCFF865QGZ.bam .bam/ENCFF438RYK.bam

# bigWig rules
# ------------------------------------------------------------------------------

$(TARGETS_RAW_ATAC): %.bw:
	$(BAMTOBIGWIG) $(BAMTOBIGWIG_ARGS_PE) \
		$(shell echo $^ | sed 's/ /,/g') $@

$(TARGETS_RPM_ATAC): %.bw:
	$(BAMTOBIGWIG) $(BAMTOBIGWIG_ARGS_PE) --normalize-track=rpm \
		$(shell echo $^ | sed 's/ /,/g') $@

$(TARGETS_RAW_HM) $(TARGETS_RAW_CTRL): %.bw:
	$(BAMTOBIGWIG) $(BAMTOBIGWIG_ARGS_SE) \
		$(shell echo $^ | sed 's/ /,/g') $@

$(TARGETS_RPM_HM) $(TARGETS_RPM_CTRL): %.bw:
	$(BAMTOBIGWIG) $(BAMTOBIGWIG_ARGS_SE) --normalize-track=rpm \
		$(shell echo $^ | sed 's/ /,/g') $@

# bam download
# ------------------------------------------------------------------------------

.bam/%.bam:
	mkdir -p .bam
	wget -O $@ "https://www.encodeproject.org/files/$(basename $(notdir $@))/@@download/$(notdir $@)"

# ------------------------------------------------------------------------------

.DELETE_ON_ERROR:
