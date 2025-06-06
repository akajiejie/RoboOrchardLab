# Minimal makefile for Sphinx dumentation
#

# You can set these variables from the command line.
SPHINXOPTS    = "-W"
# SPHINXOPTS    =
SPHINXBUILD   = sphinx-build
SPHINXPROJ    = RoboOrchardLab
SOURCEDIR     = .
BUILDDIR      = ../build/docs_build
ROBO_ORCHARD_LAB_DOCS_CURRENT_VERSION_ID ?= master
API_ROOT      = api/robo_orchard_lab

# Put it first so that "make" without argument is like "make help".
help:
	@$(SPHINXBUILD) -M help "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

# Catch-all target: route all unknown targets to Sphinx using the new
# "make mode" option.  $(O) is meant as a shortcut for $(SPHINXOPTS).
%: Makefile
	@mkdir -p $(BUILDDIR)
	@rm -rf trainer_tutorial/workspace ${API_ROOT}
	@PYTORCH_JIT=0 ROBO_ORCHARD_NO_SOURCE_FILE=1
	sphinx-apidoc -o ${API_ROOT} -q -f --separate --tocfile index ../robo_orchard_lab
	$(SPHINXBUILD) -M $@ "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)
	@cp -rf $(BUILDDIR)/html $(BUILDDIR)/$(ROBO_ORCHARD_LAB_DOCS_CURRENT_VERSION_ID)

clean:
	@rm -r $(BUILDDIR)/*
	@rm -r build/

.PHONY: help Makefile
