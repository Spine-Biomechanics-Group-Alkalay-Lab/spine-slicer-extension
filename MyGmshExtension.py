import logging
import os
from typing import Annotated, Optional
import sys
import numpy as np
import vtk
import subprocess

import slicer
from slicer.i18n import tr as _
from slicer.i18n import translate
from slicer.ScriptedLoadableModule import *
from slicer.util import VTKObservationMixin
from slicer.parameterNodeWrapper import (
    parameterNodeWrapper,
    WithinRange,
)

from slicer import vtkMRMLScalarVolumeNode


#
# MyGmshExtension
#


class MyGmshExtension(ScriptedLoadableModule):
    """Uses ScriptedLoadableModule base class, available at:
    https://github.com/Slicer/Slicer/blob/main/Base/Python/slicer/ScriptedLoadableModule.py
    """

    def __init__(self, parent):
        ScriptedLoadableModule.__init__(self, parent)
        self.parent.title = _("My Gmsh Extension")  # TODO: make this more human readable by adding spaces
        # TODO: set categories (folders where the module shows up in the module selector)
        self.parent.categories = ["Examples"]
        self.parent.dependencies = []  # TODO: add here list of module names that this module requires
        self.parent.contributors = ["Gnaneswar Chundi (BIDMC)"]  # TODO: replace with "Firstname Lastname (Organization)"
        # TODO: update with short description of the module and a link to online module documentation
        # _() function marks text as translatable to other languages
        self.parent.helpText = _("""
This is an example of scripted loadable module bundled in an extension.
See more information in <a href="https://github.com/organization/projectname#MyGmshExtension">module documentation</a>.
""")
        # TODO: replace with organization, grant and thanks
        self.parent.acknowledgementText = _("""
This file was originally developed by Jean-Christophe Fillion-Robin, Kitware Inc., Andras Lasso, PerkLab,
and Steve Pieper, Isomics, Inc. and was partially funded by NIH grant 3P41RR013218-12S1.
""")

        # Additional initialization step after application startup is complete
        #slicer.app.connect("startupCompleted()", registerSampleData)


#
# Register sample data sets in Sample Data module
#


# def registerSampleData():
#     """Add data sets to Sample Data module."""
#     # It is always recommended to provide sample data for users to make it easy to try the module,
#     # but if no sample data is available then this method (and associated startupCompeted signal connection) can be removed.

#     import SampleData

#     iconsPath = os.path.join(os.path.dirname(__file__), "Resources/Icons")

#     # To ensure that the source code repository remains small (can be downloaded and installed quickly)
#     # it is recommended to store data sets that are larger than a few MB in a Github release.

#     # MyGmshExtension1
#     SampleData.SampleDataLogic.registerCustomSampleDataSource(
#         # Category and sample name displayed in Sample Data module
#         category="MyGmshExtension",
#         sampleName="MyGmshExtension1",
#         # Thumbnail should have size of approximately 260x280 pixels and stored in Resources/Icons folder.
#         # It can be created by Screen Capture module, "Capture all views" option enabled, "Number of images" set to "Single".
#         thumbnailFileName=os.path.join(iconsPath, "MyGmshExtension1.png"),
#         # Download URL and target file name
#         uris="https://github.com/Slicer/SlicerTestingData/releases/download/SHA256/998cb522173839c78657f4bc0ea907cea09fd04e44601f17c82ea27927937b95",
#         fileNames="MyGmshExtension1.nrrd",
#         # Checksum to ensure file integrity. Can be computed by this command:
#         #  import hashlib; print(hashlib.sha256(open(filename, "rb").read()).hexdigest())
#         checksums="SHA256:998cb522173839c78657f4bc0ea907cea09fd04e44601f17c82ea27927937b95",
#         # This node name will be used when the data set is loaded
#         nodeNames="MyGmshExtension1",
#     )

#     # MyGmshExtension2
#     SampleData.SampleDataLogic.registerCustomSampleDataSource(
#         # Category and sample name displayed in Sample Data module
#         category="MyGmshExtension",
#         sampleName="MyGmshExtension2",
#         thumbnailFileName=os.path.join(iconsPath, "MyGmshExtension2.png"),
#         # Download URL and target file name
#         uris="https://github.com/Slicer/SlicerTestingData/releases/download/SHA256/1a64f3f422eb3d1c9b093d1a18da354b13bcf307907c66317e2463ee530b7a97",
#         fileNames="MyGmshExtension2.nrrd",
#         checksums="SHA256:1a64f3f422eb3d1c9b093d1a18da354b13bcf307907c66317e2463ee530b7a97",
#         # This node name will be used when the data set is loaded
#         nodeNames="MyGmshExtension2",
#     )


#
# MyGmshExtensionParameterNode
#


# @parameterNodeWrapper
# class MyGmshExtensionParameterNode:
#     """
#     The parameters needed by module.

#     inputVolume - The volume to threshold.
#     imageThreshold - The value at which to threshold the input volume.
#     invertThreshold - If true, will invert the threshold.
#     thresholdedVolume - The output volume that will contain the thresholded volume.
#     invertedVolume - The output volume that will contain the inverted thresholded volume.
#     """

#     inputModel: slicer.vtkMRMLModelNode
#     #inputVolume: vtkMRMLScalarVolumeNode
#     imageThreshold: Annotated[float, WithinRange(-100, 500)] = 100
#     invertThreshold: bool = False
#     thresholdedVolume: vtkMRMLScalarVolumeNode
#     invertedVolume: vtkMRMLScalarVolumeNode
#     elementSize: float = 1.0
#     optimizeNetgen: bool = True


#
# MyGmshExtensionWidget
#


class MyGmshExtensionWidget(ScriptedLoadableModuleWidget, VTKObservationMixin):
    """Uses ScriptedLoadableModuleWidget base class, available at:
    https://github.com/Slicer/Slicer/blob/main/Base/Python/slicer/ScriptedLoadableModule.py
    """

    def __init__(self, parent=None) -> None:
        """Called when the user opens the module the first time and the widget is initialized."""
        ScriptedLoadableModuleWidget.__init__(self, parent)
        VTKObservationMixin.__init__(self)  # needed for parameter node observation
        self.logic = None
        self._parameterNode = None
        self._parameterNodeGuiTag = None ###
        self._updatingGUIFromParameterNode = False  # Add this line


    def setup(self) -> None: ###
        """Called when the user opens the module the first time and the widget is initialized."""
        ScriptedLoadableModuleWidget.setup(self)

        # Load widget from .ui file (created by Qt Designer).
        # Additional widgets can be instantiated manually and added to self.layout.
        uiWidget = slicer.util.loadUI(self.resourcePath("UI/MyGmshExtension.ui"))
        self.layout.addWidget(uiWidget)
        self.ui = slicer.util.childWidgetVariables(uiWidget)

        # Set scene in MRML widgets. Make sure that in Qt designer the top-level qMRMLWidget's
        # "mrmlSceneChanged(vtkMRMLScene*)" signal in is connected to each MRML widget's.
        # "setMRMLScene(vtkMRMLScene*)" slot.
        uiWidget.setMRMLScene(slicer.mrmlScene)

        # Create logic class. Logic implements all computations that should be possible to run
        # in batch mode, without a graphical user interface.
        self.logic = MyGmshExtensionLogic()

        # Connections

        # These connections ensure that we update parameter node when scene is closed
        self.addObserver(slicer.mrmlScene, slicer.mrmlScene.StartCloseEvent, self.onSceneStartClose)
        self.addObserver(slicer.mrmlScene, slicer.mrmlScene.EndCloseEvent, self.onSceneEndClose)

        self.ui.inputVolumeSelector.nodeTypes = ["vtkMRMLModelNode"]
        self.ui.inputVolumeSelector.setMRMLScene(slicer.mrmlScene)
        self.ui.inputVolumeSelector.setToolTip("Select the input model for mesh generation")
        self.ui.ctVolumeSelector.setMRMLScene(slicer.mrmlScene)
        
        self.ui.inputVolumeSelector.connect("currentNodeChanged(vtkMRMLNode*)", self.updateParameterNodeFromGUI)
        self.ui.outputDirectorySelector.connect("directoryChanged(QString)", self.updateParameterNodeFromGUI)
        self.ui.elementSizeSpinBox.connect("valueChanged(double)", self.updateParameterNodeFromGUI)
        self.ui.optimizeNetgenCheckBox.connect("toggled(bool)", self.updateParameterNodeFromGUI)
        self.ui.ctVolumeSelector.connect("currentNodeChanged(vtkMRMLNode*)", self.updateParameterNodeFromGUI)
        self.ui.enableClippingCheckBox.connect("toggled(bool)", self.onClippingToggled)

        # Buttons
        self.ui.applyButton.connect("clicked(bool)", self.onApplyButton)

        # Make sure parameter node is initialized (needed for module reload)
        self.initializeParameterNode()

    def onClippingToggled(self, checked):
        if self.logic.outputMeshNode:
            displayNode = self.logic.outputMeshNode.GetDisplayNode()
            if displayNode:
                displayNode.SetClipping(checked)

    def cleanup(self) -> None:
        """Called when the application closes and the module widget is destroyed."""
        self.removeObservers()

    def enter(self) -> None:
        """Called each time the user opens this module."""
        # Make sure parameter node exists and observed
        self.initializeParameterNode()

    def exit(self) -> None:
        """Called each time the user opens a different module."""
        self.removeObserver(self._parameterNode, vtk.vtkCommand.ModifiedEvent, self.updateGUIFromParameterNode)
        # Do not react to parameter node changes (GUI will be updated when the user enters into the module)
        # if self._parameterNode:
        #     self._parameterNode.disconnectGui(self._parameterNodeGuiTag)
        #     self._parameterNodeGuiTag = None
        #     self.removeObserver(self._parameterNode, vtk.vtkCommand.ModifiedEvent, self._checkCanApply)

    def onSceneStartClose(self, caller, event) -> None:
        """Called just before the scene is closed."""
        # Parameter node will be reset, do not use it anymore
        self.setParameterNode(None)

    def onSceneEndClose(self, caller, event) -> None:
        """Called just after the scene is closed."""
        # If this module is shown while the scene is closed then recreate a new parameter node immediately
        if self.parent.isEntered:
            self.initializeParameterNode()

    def initializeParameterNode(self) -> None:
        """Ensure parameter node exists and observed."""
        # Parameter node stores all user choices in parameter values, node selections, etc.
        # so that when the scene is saved and reloaded, these settings are restored.

        self.setParameterNode(self.logic.getParameterNode())

        # Select default input nodes if nothing is selected yet to save a few clicks for the user
        if not self._parameterNode.GetNodeReference("InputModel"):
            firstVolumeNode = slicer.mrmlScene.GetFirstNodeByClass("vtkMRMLScalarModelNode")
            if firstVolumeNode:
                self._parameterNode.SetNodeReferenceID("InputModel", firstVolumeNode.GetID())

    def setParameterNode(self, inputParameterNode): 
        """
        Set and observe parameter node.
        Observation is needed because when the parameter node is changed then the GUI must be updated immediately.
        """
        #if inputParameterNode:
        #    self.logic.setDefaultParameters(inputParameterNode)

        if self._parameterNode is not None:
            self.removeObserver(self._parameterNode, vtk.vtkCommand.ModifiedEvent, self.updateGUIFromParameterNode)
        self._parameterNode = inputParameterNode
        if self._parameterNode is not None:
            self.addObserver(self._parameterNode, vtk.vtkCommand.ModifiedEvent, self.updateGUIFromParameterNode)

        # Initial GUI update
        self.updateGUIFromParameterNode()

    def updateGUIFromParameterNode(self, caller=None, event=None):
        """
        This method is called whenever parameter node is changed.
        The module GUI is updated to show the current state of the parameter node.
        """

        if self._parameterNode is None or self._updatingGUIFromParameterNode:
            return

        # Make sure GUI changes do not call updateParameterNodeFromGUI (it could cause infinite loop)
        self._updatingGUIFromParameterNode = True

        # Update node selectors and sliders
        self.ui.inputVolumeSelector.setCurrentNode(self._parameterNode.GetNodeReference("InputModel"))
        self.ui.ctVolumeSelector.setCurrentNode(self._parameterNode.GetNodeReference("CTVolume"))
        self.ui.outputDirectorySelector.directory = self._parameterNode.GetParameter("OutputDirectory")
        self.ui.elementSizeSpinBox.value = float(self._parameterNode.GetParameter("ElementSize"))
        self.ui.optimizeNetgenCheckBox.checked = self._parameterNode.GetParameter("OptimizeNetgen") == "true"
        #self.ui.inputSeedSelector.setCurrentNode(self._parameterNode.GetNodeReference("InputSeed"))
        #self.ui.outputSegmentationSelector.setCurrentNode(self._parameterNode.GetNodeReference("OutputSegmentation"))

        # Update buttons states and tooltips
        if self._parameterNode.GetNodeReference("InputModel") and self._parameterNode.GetParameter("OutputDirectory"):
            self.ui.applyButton.toolTip = "Compute mesh"
            self.ui.applyButton.enabled = True
        else:
            self.ui.applyButton.toolTip = "Select input segmentation volume"
            self.ui.applyButton.enabled = False

        # All the GUI updates are done
        self._updatingGUIFromParameterNode = False
    
    def updateParameterNodeFromGUI(self, caller=None, event=None):
        """
        This method is called when the user makes any change in the GUI.
        The changes are saved into the parameter node (so that they are restored when the scene is saved and loaded).
        """

        if self._parameterNode is None or self._updatingGUIFromParameterNode:
            return

        wasModified = self._parameterNode.StartModify()  # Modify all properties in a single batch

        self._parameterNode.SetNodeReferenceID("InputModel", self.ui.inputVolumeSelector.currentNodeID)
        self._parameterNode.SetNodeReferenceID("CTVolume", self.ui.ctVolumeSelector.currentNodeID)

        self._parameterNode.SetParameter("OutputDirectory", self.ui.outputDirectorySelector.directory)
        self._parameterNode.SetParameter("ElementSize", str(self.ui.elementSizeSpinBox.value))
        self._parameterNode.SetParameter("OptimizeNetgen", "true" if self.ui.optimizeNetgenCheckBox.checked else "false") ###
        self._parameterNode.EndModify(wasModified)
        #self._parameterNode.SetNodeReferenceID("InputSeed", self.ui.inputSeedSelector.currentNodeID)
        #self._parameterNode.SetNodeReferenceID("OutputVolume", self.ui.outputSegmentationSelector.currentNodeID)

        self._parameterNode.EndModify(wasModified)
    
    def onApplyButton(self):
        """
        Run processing when user clicks "Apply" button.
        """
        with slicer.util.tryWithErrorDisplay("Failed to compute results.", waitCursor=True):
            inputModel = self.ui.inputVolumeSelector.currentNode()
            outputDirectory = self.ui.outputDirectorySelector.directory
            elementSize = self.ui.elementSizeSpinBox.value
            optimizeNetgen = self.ui.optimizeNetgenCheckBox.checked
            ctVolume = self.ui.ctVolumeSelector.currentNode()
            if not inputModel or not outputDirectory or not ctVolume:
                raise ValueError("Input model, output directory, or CT volume is invalid")
            meshedModelNode = self.logic.process(inputModel, outputDirectory, elementSize, optimizeNetgen, ctVolume)
        
            if meshedModelNode:
                slicer.util.setSliceViewerLayers(background=None)
                slicer.util.resetSliceViews()
                slicer.app.layoutManager().setLayout(slicer.vtkMRMLLayoutNode.SlicerLayoutOneUp3DView)
                slicer.util.resetThreeDViews()
                
                # Set the meshed model to be visible
                meshedModelNode.GetDisplayNode().SetVisibility(True)

        #self.ui.applyButton.enabled = True
        #convolutionKernel = self.logic.convolutionKernelFromVolumeNode(inputVolume)
        # if not convolutionKernel:
        #     if not slicer.util.confirmOkCancelDisplay("Convolution kernel cannot be determined from the input volume."
        #         " The current input volume is not loaded from DICOM or the Convolution Kernel (0018,1210) field is missing."
        #         " Click OK to use STANDARD convolution kernel.",
        #         dontShowAgainSettingsKey = "AirwaySegmentation/DontShowDICOMImageExpectedWarning"):
        #         return False

    # def _checkCanApply(self, caller=None, event=None) -> None:
    #     if self._parameterNode and self._parameterNode.inputVolume and self._parameterNode.thresholdedVolume:
    #         self.ui.applyButton.toolTip = _("Compute output volume")
    #         self.ui.applyButton.enabled = True
    #     else:
    #         self.ui.applyButton.toolTip = _("Select input and output volume nodes")
    #         self.ui.applyButton.enabled = False

    # def onApplyButton(self) -> None:
    #     """Run processing when user clicks "Apply" button."""
    #     with slicer.util.tryWithErrorDisplay(_("Failed to compute results."), waitCursor=True):
    #         # Compute output
    #         self.logic.process(self.ui.inputSelector.currentNode(), self.ui.outputSelector.currentNode(),
    #                            self.ui.imageThresholdSliderWidget.value, self.ui.invertOutputCheckBox.checked)

    #         # Compute inverted output (if needed)
    #         if self.ui.invertedOutputSelector.currentNode():
    #             # If additional output volume is selected then result with inverted threshold is written there
    #             self.logic.process(self.ui.inputSelector.currentNode(), self.ui.invertedOutputSelector.currentNode(),
    #                                self.ui.imageThresholdSliderWidget.value, not self.ui.invertOutputCheckBox.checked, showResult=False)


#
# MyGmshExtensionLogic
#


class MyGmshExtensionLogic(ScriptedLoadableModuleLogic):
    """This class should implement all the actual
    computation done by your module.  The interface
    should be such that other python code can import
    this class and make use of the functionality without
    requiring an instance of the Widget.
    Uses ScriptedLoadableModuleLogic base class, available at:
    https://github.com/Slicer/Slicer/blob/main/Base/Python/slicer/ScriptedLoadableModule.py
    """

    def __init__(self) -> None:
        """Called when the logic class is instantiated. Can be used for initializing member variables."""
        ScriptedLoadableModuleLogic.__init__(self)
        self.outputMeshNode = None

    # def setDefaultParameters(self, parameterNode):
    #     if not parameterNode.GetParameter("ElementSize"):
    #         parameterNode.SetParameter("ElementSize", "1.0")

    def process(self, inputModel, outputDirectory, elementSize, optimizeNetgen, ctVolume):
        if not inputModel or not outputDirectory or not ctVolume:
            raise ValueError("Input model, output path, or CT volume is invalid")
        
        if not inputModel.IsA("vtkMRMLModelNode"):
            raise ValueError("Input must be a surface model (vtkMRMLModelNode)")
        
        # polyData = inputModel.GetPolyData()
        # if not polyData or polyData.GetNumberOfPoints() == 0:
        #     raise ValueError("Input model does not contain valid geometry")

        os.makedirs(outputDirectory, exist_ok=True)

        # # Save input volume as VTK file
        # tempVtkPath = os.path.join(outputDirectory, "temp_input.vtk")
        # writer = vtk.vtkPolyDataWriter()
        # writer.SetInputData(inputModel.GetPolyData())
        # writer.SetFileName(tempVtkPath)
        # writer.SetFileTypeToASCII()
        # writer.Write()

        # if not os.path.exists(tempVtkPath):
        #     raise ValueError(f"Failed to create VTK file: {tempVtkPath}")

        # Calculate Young's modulus based on CT data
        youngModulusArray = self.calculateYoungModulus(inputModel, ctVolume)

        # Export the model to VTK file
        exportVtkPath = os.path.join(outputDirectory, "exported_model.vtk")
        slicer.util.saveNode(inputModel, exportVtkPath, properties={"useCompression": False})

        if not os.path.exists(exportVtkPath):
            raise ValueError(f"Failed to create VTK file: {exportVtkPath}")

        print(f"VTK file created: {exportVtkPath}")
        print(f"VTK file size: {os.path.getsize(exportVtkPath)} bytes")

        # Print the first few lines of the VTK file
        with open(exportVtkPath, 'r') as f:
            print("First few lines of VTK file:")
            print(f.read(500))

        output_vtk_file = self.generateMesh(exportVtkPath, outputDirectory, elementSize, optimizeNetgen)

        # Load the generated VTK file back into Slicer
        #loadedModelNode = slicer.util.loadModel(output_vtk_file)
        outputMeshNode = self.loadMeshIntoSlicer(output_vtk_file)
        if outputMeshNode:
            print(f"Loaded meshed model: {outputMeshNode.GetName()}")
            # Optionally, you can set a custom name for the loaded model
            outputMeshNode.SetName("Meshed_Model")
            self.assignMaterialProperties(outputMeshNode, ctVolume)

            # Save the mesh as a .summit file
            summitFilePath = os.path.join(outputDirectory, "meshed_model.summit")
            self.saveSummitFile(outputMeshNode, summitFilePath)

            return outputMeshNode
        else:
            print("Failed to load the meshed model")
        

        # # Load the generated mesh into Slicer
        # meshedModelNode = self.loadMeshIntoSlicer(output_vtk_file)
        
        # if meshedModelNode:
        #     # Assign material properties
        #     self.assignMaterialProperties(meshedModelNode, ctVolume)
            
        #     # Set up visualization (existing code)
        #     # ...
        # else:
        #     print("NO MODEL FOUND")

        # return meshedModelNode

    def saveSummitFile(self, meshNode, outputPath):
        if not meshNode or not outputPath:
            logging.error("Invalid mesh node or output path")
            return

        mesh = meshNode.GetMesh()
        if not mesh:
            logging.error("No mesh data found")
            return

        nNodes = mesh.GetNumberOfPoints()
        nElems = mesh.GetNumberOfCells()

        with open(outputPath, 'w') as f:
            # Write header
            f.write("3\n")
            f.write(f"{nNodes} {nElems} 1 1\n")

            # Write node coordinates
            for i in range(nNodes):
                point = mesh.GetPoint(i)
                f.write(f"{point[0]:.15f} {point[1]:.15f} {point[2]:.15f}\n")

            # Write element connectivity
            for i in range(nElems):
                cell = mesh.GetCell(i)
                pointIds = [str(cell.GetPointId(j)) for j in range(cell.GetNumberOfPoints())]
                f.write(f"1 {' '.join(pointIds)}\n")

            # Write number of internal variables
            f.write("10\n")

            # Write BVTV (Young's modulus) values
            youngModulusArray = mesh.GetCellData().GetArray("YoungModulus")
            if youngModulusArray:
                for i in range(nElems):
                    f.write(f"{youngModulusArray.GetValue(i):.15f}\n")
            else:
                logging.warning("Young's modulus data not found. Writing default values.")
                for _ in range(nElems):
                    f.write("1.0\n")

        logging.info(f"Summit file saved: {outputPath}")

    def assignMaterialProperties(self, meshedModelNode, ctVolume):
        # Get the unstructured grid from the meshed model
        meshData = meshedModelNode.GetMesh()
        
        # Get CT image data
        ctImageData = ctVolume.GetImageData()
        ctArray = slicer.util.array(ctVolume.GetID())

        # Get CT volume to RAS matrix
        rasToIJK = vtk.vtkMatrix4x4()
        ctVolume.GetRASToIJKMatrix(rasToIJK)

        # Create a new array for Young's modulus
        youngModulusArray = vtk.vtkDoubleArray()
        youngModulusArray.SetName("YoungModulus")
        youngModulusArray.SetNumberOfComponents(1)
        youngModulusArray.SetNumberOfTuples(meshData.GetNumberOfCells())

        # Base Young's modulus
        baseYoungModulus = 1.5

        # Iterate over cells (tetrahedra)
        for i in range(meshData.GetNumberOfCells()):
            cell = meshData.GetCell(i)
            centroid = [0, 0, 0]
            for j in range(cell.GetNumberOfPoints()):
                point = cell.GetPoints().GetPoint(j)
                centroid[0] += point[0]
                centroid[1] += point[1]
                centroid[2] += point[2]
            centroid = [x / cell.GetNumberOfPoints() for x in centroid]

            # Transform centroid to IJK coordinates
            centroidIJK = rasToIJK.MultiplyPoint(centroid + [1])[:3]
            centroidIJK = [int(round(x)) for x in centroidIJK]

            # Get CT value at centroid
            if (0 <= centroidIJK[0] < ctArray.shape[2] and
                0 <= centroidIJK[1] < ctArray.shape[1] and
                0 <= centroidIJK[2] < ctArray.shape[0]):
                ctValue = ctArray[centroidIJK[2], centroidIJK[1], centroidIJK[0]]
                
                # Adjust Young's modulus based on CT value
                youngModulus = baseYoungModulus
                if ctValue > np.mean(ctArray):
                    youngModulus *= 1 + 2 * (ctValue - np.mean(ctArray)) / (np.max(ctArray) - np.mean(ctArray))
                
                youngModulusArray.SetValue(i, youngModulus)
            else:
                youngModulusArray.SetValue(i, baseYoungModulus)

        # Add the Young's modulus array to the mesh
        meshData.GetCellData().AddArray(youngModulusArray)
        meshedModelNode.Modified()

        # Log statistics
        youngModulusMin, youngModulusMax = youngModulusArray.GetRange()
        print(f"Material properties assigned.")
        print(f"Young's modulus range: {youngModulusMin:.2f} - {youngModulusMax:.2f}")
        print(f"Mean Young's modulus: {np.mean(youngModulusArray):.2f}")
        print(f"Median Young's modulus: {np.median(youngModulusArray):.2f}")

        # Set up color mapping for visualization
        displayNode = meshedModelNode.GetDisplayNode()
        displayNode.SetActiveScalarName("YoungModulus")
        displayNode.SetAndObserveColorNodeID("vtkMRMLColorTableNodeRainbow")
        displayNode.SetScalarVisibility(True)
        displayNode.SetScalarRangeFlag(slicer.vtkMRMLDisplayNode.UseDataScalarRange)

    def calculateYoungModulus(self, inputModel, ctVolume):
        polyData = inputModel.GetPolyData()
        points = polyData.GetPoints()

        # Get CT image data
        ctImageData = ctVolume.GetImageData()
        ctArray = slicer.util.array(ctVolume.GetID())

        # Get CT volume to RAS matrix
        rasToIJK = vtk.vtkMatrix4x4()
        ctVolume.GetRASToIJKMatrix(rasToIJK)

        youngModulusArray = np.ones(polyData.GetNumberOfCells()) * 1.5  # Base Young's modulus

        cellIds = vtk.vtkIdList()
        for i in range(polyData.GetNumberOfCells()):
            polyData.GetCellPoints(i, cellIds)
            centroid = [0, 0, 0]
            for j in range(cellIds.GetNumberOfIds()):
                point = points.GetPoint(cellIds.GetId(j))
                centroid[0] += point[0]
                centroid[1] += point[1]
                centroid[2] += point[2]
            centroid = [x / cellIds.GetNumberOfIds() for x in centroid]

            # Transform centroid to IJK coordinates
            centroidIJK = rasToIJK.MultiplyPoint(centroid + [1])[:3]
            centroidIJK = [int(round(x)) for x in centroidIJK]

            # Get CT value at centroid
            if (0 <= centroidIJK[0] < ctArray.shape[2] and
                0 <= centroidIJK[1] < ctArray.shape[1] and
                0 <= centroidIJK[2] < ctArray.shape[0]):
                ctValue = ctArray[centroidIJK[2], centroidIJK[1], centroidIJK[0]]
                
                # Adjust Young's modulus based on CT value
                if ctValue > np.mean(ctArray):
                    youngModulusArray[i] *= 3 * (ctValue - np.mean(ctArray)) / (np.max(ctArray) - np.mean(ctArray))

        return youngModulusArray

    def loadMeshIntoSlicer(self, vtk_file_path):
        # Read the VTK file
        reader = vtk.vtkUnstructuredGridReader()
        reader.SetFileName(vtk_file_path)
        reader.Update()

         # Create a new model node for the mesh
        outputMeshNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLModelNode")
        outputMeshNode.SetName("GMSH_Output")

        # Get the mesh data
        meshData = reader.GetOutput()

        # Create a transform to flip X and Y coordinates
        transform = vtk.vtkTransform()
        transform.Scale(-1, -1, 1)  # Flip X and Y, keep Z the same

        # Apply the transform to the mesh
        transformFilter = vtk.vtkTransformFilter()
        transformFilter.SetInputData(meshData)
        transformFilter.SetTransform(transform)
        transformFilter.Update()

        # Set the transformed mesh data
        outputMeshNode.SetAndObserveMesh(transformFilter.GetOutput())

        # Set the mesh data
        outputMeshNode.SetAndObserveMesh(reader.GetOutput())

        # Create and configure the display node
        outputMeshNode.CreateDefaultDisplayNodes()
        outputMeshDisplayNode = outputMeshNode.GetDisplayNode()
        outputMeshDisplayNode.SetEdgeVisibility(True)
        outputMeshDisplayNode.SetClipping(True)

        # Set up color mapping
        colorTableNode = slicer.util.getNode('GenericAnatomyColors')
        if not colorTableNode:
            colorTableNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLColorTableNode")
            colorTableNode.SetTypeToGenericAnatomyColors()
            slicer.util.saveNode(colorTableNode, os.path.join(slicer.app.temporaryPath, "GenericAnatomyColors.ctbl"))
        outputMeshDisplayNode.SetAndObserveColorNodeID(colorTableNode.GetID())

        # Set up clipping planes
        sliceNodes = slicer.util.getNodesByClass('vtkMRMLSliceNode')
        for sliceNode in sliceNodes:
            clipNode = slicer.mrmlScene.GetFirstNodeByClass("vtkMRMLClipModelsNode")
            if not clipNode:
                clipNode = slicer.vtkMRMLClipModelsNode()
                slicer.mrmlScene.AddNode(clipNode)
            
            if sliceNode.GetName() == 'Red':
                clipNode.SetRedSliceClipState(clipNode.ClipNegativeSpace)
            elif sliceNode.GetName() == 'Green':
                clipNode.SetGreenSliceClipState(clipNode.ClipNegativeSpace)
            elif sliceNode.GetName() == 'Yellow':
                clipNode.SetYellowSliceClipState(clipNode.ClipNegativeSpace)

        # Configure scalar visibility and attributes
        outputMeshDisplayNode.ScalarVisibilityOn()
        
        # Check if 'labels' exists in the mesh data, otherwise use the first available array
        pointData = reader.GetOutput().GetPointData()
        cellData = reader.GetOutput().GetCellData()
        if cellData.HasArray('labels'):
            outputMeshDisplayNode.SetActiveScalarName('labels')
            outputMeshDisplayNode.SetActiveAttributeLocation(vtk.vtkAssignAttribute.CELL_DATA)
        elif pointData.HasArray('labels'):
            outputMeshDisplayNode.SetActiveScalarName('labels')
            outputMeshDisplayNode.SetActiveAttributeLocation(vtk.vtkAssignAttribute.POINT_DATA)
        elif cellData.GetNumberOfArrays() > 0:
            outputMeshDisplayNode.SetActiveScalarName(cellData.GetArrayName(0))
            outputMeshDisplayNode.SetActiveAttributeLocation(vtk.vtkAssignAttribute.CELL_DATA)
        elif pointData.GetNumberOfArrays() > 0:
            outputMeshDisplayNode.SetActiveScalarName(pointData.GetArrayName(0))
            outputMeshDisplayNode.SetActiveAttributeLocation(vtk.vtkAssignAttribute.POINT_DATA)
        else:
            print("No scalar data found in the mesh")
            outputMeshDisplayNode.ScalarVisibilityOff()

        outputMeshDisplayNode.SetVisibility2D(True)
        outputMeshDisplayNode.SetSliceIntersectionOpacity(0.5)
        outputMeshDisplayNode.SetScalarRangeFlag(slicer.vtkMRMLDisplayNode.UseColorNodeScalarRange)

        self.outputMeshNode = outputMeshNode

        return outputMeshNode

        # print(f"VTK file created: {tempVtkPath}")
        # print(f"VTK file size: {os.path.getsize(tempVtkPath)} bytes")

        # # Print the first few lines of the VTK file
        # with open(tempVtkPath, 'r') as f:
        #     print("First few lines of VTK file:")
        #     print(f.read(500))

        # Generate mesh
        #self.generateMesh(exportVtkPath, outputDirectory)

        # Clean up temporary file
        #os.remove(tempVtkPath)
    
    def verifyAlignment(self, meshNode, inputVolumeNode):
        # Get the bounds of the mesh
        meshBounds = meshNode.GetPolyData().GetBounds()

        # Get the bounds of the input volume
        volumeBounds = [0] * 6
        inputVolumeNode.GetRASBounds(volumeBounds)

        print("Mesh bounds (RAS):", meshBounds)
        print("Volume bounds (RAS):", volumeBounds)

        # Calculate the center of the mesh
        meshCenter = [
            (meshBounds[1] + meshBounds[0]) / 2,
            (meshBounds[3] + meshBounds[2]) / 2,
            (meshBounds[5] + meshBounds[4]) / 2
        ]

        # Calculate the center of the volume
        volumeCenter = [
            (volumeBounds[1] + volumeBounds[0]) / 2,
            (volumeBounds[3] + volumeBounds[2]) / 2,
            (volumeBounds[5] + volumeBounds[4]) / 2
        ]

        print("Mesh center (RAS):", meshCenter)
        print("Volume center (RAS):", volumeCenter)

        # Calculate the difference
        centerDifference = [
            meshCenter[0] - volumeCenter[0],
            meshCenter[1] - volumeCenter[1],
            meshCenter[2] - volumeCenter[2]
        ]

        print("Center difference (RAS):", centerDifference)

        return centerDifference
    
    def generateMesh(self, vtkFilePath, outputDirectory, elementSize, optimizeNetgen):
        print(f"Input VTK file: {vtkFilePath}")
        print(f"Output directory: {outputDirectory}")

        base_name = os.path.splitext(os.path.basename(vtkFilePath))[0]
        geo_file = os.path.join(outputDirectory, base_name + '.geo')
        output_vtk_file = os.path.join(outputDirectory, base_name + '_meshed.vtk')

        print(f"GEO file: {geo_file}")
        print(f"Output VTK file: {output_vtk_file}")

        self.create_geo_file(vtkFilePath, geo_file, element_size=elementSize)
        self.generate_mesh_with_gmsh(geo_file, output_vtk_file, elementSize, optimizeNetgen)

        return output_vtk_file

    def create_geo_file(self, vtk_file_name, geo_file, element_size):
        """
        Create a GEO file for GMSH from a VTK file.

        Parameters:
        vtk_file_name (str): Path to the input VTK file.
        geo_file (str): Path to the output GEO file.
        element_size (float): Desired element size for the mesh.
        """
        # Open the GEO file for writing
        with open(geo_file, 'w') as file:
            #file.write(f'SetFactory("OpenCASCADE");\n')
            # Merge the VTK file into the GEO file
            file.write(f'Merge "{vtk_file_name}";\n')
            # Create a surface loop with ID 1
            file.write("Surface Loop(1) = {1};\n")
            # Create a volume with ID 1 based on the surface loop
            file.write("Volume(1) = {1};\n")
            # Set the characteristic length (element size) for the points in the volume
            file.write(f'Characteristic Length {{ PointsOf {{ Volume {{1}} }} }} = {element_size};\n')
            # Assign the volume a physical name "myVolume"
            file.write('Physical Volume("myVolume") = {1};\n')
        
        print(f"GEO file created: {geo_file}")
        with open(geo_file, 'r') as file:
            print("GEO file contents:")
            print(file.read())

    def getParameterNode(self):
        return ScriptedLoadableModuleLogic.getParameterNode(self)

    ##gmsh executable 

    def generate_mesh_with_gmsh(self, geo_file, output_file, elementSize, optimizeNetgen):
        current_dir = os.path.dirname(os.path.abspath(__file__))
        gmsh_script = os.path.join(current_dir, "gmsh")
        
        # Use Slicer's Python interpreter
        python_executable = sys.executable

        print(f"GMSH script path: {gmsh_script}")
        print(f"Python executable: {python_executable}")
        print(f"GEO file path: {geo_file}")
        print(f"MSH file path: {output_file}")

        gmsh_command = [
            python_executable, gmsh_script, 
            geo_file, 
            '-3',  # 3D mesh
            '-order', '2',  # Second order elements
            f'-clmax', str(elementSize),  # Set element size
            '-format', 'vtk', 
            '-o', output_file
        ]
        
        if optimizeNetgen:
            gmsh_command.append('-optimize_netgen')

        try:
            result = subprocess.run(gmsh_command, check=True, capture_output=True, text=True,
                                    env=dict(os.environ, PYTHONPATH=current_dir))
            print("GMSH stdout:")
            print(result.stdout)
        except subprocess.CalledProcessError as e:
            print(f"Error during GMSH execution (return code {e.returncode}):")
            print("STDOUT:")
            print(e.stdout)
            print("STDERR:")
            print(e.stderr)
            raise
        except Exception as e:
            print(f"Unexpected error: {str(e)}")
            raise

        if os.path.exists(output_file):
            print(f"Output file created successfully: {output_file}")
            print(f"Output file size: {os.path.getsize(output_file)} bytes")
        else:
            print(f"Failed to create output file: {output_file}")

#
# MyGmshExtensionTest
#


class MyGmshExtensionTest(ScriptedLoadableModuleTest):
    """
    This is the test case for your scripted module.
    Uses ScriptedLoadableModuleTest base class, available at:
    https://github.com/Slicer/Slicer/blob/main/Base/Python/slicer/ScriptedLoadableModule.py
    """

    def setUp(self):
        """Do whatever is needed to reset the state - typically a scene clear will be enough."""
        slicer.mrmlScene.Clear()

    def runTest(self):
        """Run as few or as many tests as needed here."""
        self.setUp()
        self.test_MyGmshExtension1()

    def test_MyGmshExtension1(self):
        """Ideally you should have several levels of tests.  At the lowest level
        tests should exercise the functionality of the logic with different inputs
        (both valid and invalid).  At higher levels your tests should emulate the
        way the user would interact with your code and confirm that it still works
        the way you intended.
        One of the most important features of the tests is that it should alert other
        developers when their changes will have an impact on the behavior of your
        module.  For example, if a developer removes a feature that you depend on,
        your test should break so they know that the feature is needed.
        """

        self.delayDisplay("Starting the test")

        # Get/create input data

        import SampleData

        registerSampleData()
        inputVolume = SampleData.downloadSample("MyGmshExtension1")
        self.delayDisplay("Loaded test data set")

        inputScalarRange = inputVolume.GetImageData().GetScalarRange()
        self.assertEqual(inputScalarRange[0], 0)
        self.assertEqual(inputScalarRange[1], 695)

        outputVolume = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLScalarVolumeNode")
        threshold = 100

        # Test the module logic

        logic = MyGmshExtensionLogic()

        # Test algorithm with non-inverted threshold
        logic.process(inputVolume, outputVolume, threshold, True)
        outputScalarRange = outputVolume.GetImageData().GetScalarRange()
        self.assertEqual(outputScalarRange[0], inputScalarRange[0])
        self.assertEqual(outputScalarRange[1], threshold)

        # Test algorithm with inverted threshold
        logic.process(inputVolume, outputVolume, threshold, False)
        outputScalarRange = outputVolume.GetImageData().GetScalarRange()
        self.assertEqual(outputScalarRange[0], inputScalarRange[0])
        self.assertEqual(outputScalarRange[1], inputScalarRange[1])

        self.delayDisplay("Test passed")
