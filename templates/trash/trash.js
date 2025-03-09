class DiagramModal {
    constructor() {
        this.createModal();
        this.zoomLevel = 1;
        this.isDragging = false;
        this.currentX = 0;
        this.currentY = 0;
        this.initialX = 0;
        this.initialY = 0;
        this.xOffset = 0;
        this.yOffset = 0;
        
        // Multi-touch zoom variables
        this.initialDistance = 0;
        this.currentDistance = 0;
    }
    createModal() {
        // Create modal structure - now medium screen size
        this.modal = document.createElement('div');
        this.modal.className = 'fixed inset-0 bg-black bg-opacity-50 z-50 flex items-center justify-center p-4';
        
        // Modal content - reduced to about 85% of screen size
        this.modalContent = document.createElement('div');
        this.modalContent.className = 'bg-white dark:bg-gray-800 rounded-xl shadow-2xl p-4 w-full h-[75vh] overflow-hidden relative';
                
        // Rest of the modal creation remains similar to previous implementation
        this.modalHeader = document.createElement('div');
        this.modalHeader.className = 'flex justify-between items-center mb-4';
        
        this.modalTitle = document.createElement('h3');
        this.modalTitle.className = 'text-lg font-semibold text-gray-800 dark:text-gray-200';
        this.modalTitle.textContent = 'Diagram View';
        
        this.controlsContainer = document.createElement('div');
        this.controlsContainer.className = 'absolute top-4 right-4 flex items-center space-x-2 z-10';

        this.modal.addEventListener('click', (e) => {
            // Close modal if clicked outside modalContent
            if (!this.modalContent.contains(e.target)) {
                this.close();
            }
        });

        // Zoom controls
        this.zoomInBtn = this.createIconButton('➕', this.zoomIn.bind(this), 'Zoom In');
        this.zoomOutBtn = this.createIconButton('➖', this.zoomOut.bind(this), 'Zoom Out');
        this.zoomResetBtn = this.createIconButton('↩️', this.resetZoom.bind(this), 'Reset Zoom');
        this.closeBtn = this.createIconButton('✖️', this.close.bind(this), 'Close');
        this.downloadBtn = this.createIconButton('⬇️', this.downloadDiagram.bind(this), 'Download Diagram');
        
        this.controlsContainer.append(
            this.zoomInBtn, 
            this.zoomOutBtn, 
            this.zoomResetBtn, 
            this.downloadBtn, 
            this.closeBtn
        );
        
        this.modalHeader.append(this.modalTitle, this.controlsContainer);
        
        // Body - now with slightly reduced height
        this.modalBody = document.createElement('div');
        this.modalBody.className = 'relative w-full h-[calc(85vh-100px)] overflow-auto';
        
        // Diagram container with improved touch support
        this.diagramContainer = document.createElement('div');
        this.diagramContainer.className = 'diagram-container w-full h-full overflow-auto touch-none';
        
        this.modalBody.appendChild(this.diagramContainer);
        this.modalContent.append(this.modalHeader, this.modalBody);
        this.modal.appendChild(this.modalContent);
        
        // Initially hidden
        this.modal.style.display = 'none';
        
        document.body.appendChild(this.modal);
        
        // Enhanced touch and mouse interactions
        this.setupTouchZoom();
        this.setupMouseZoom();
    }

    setupTouchZoom() {
        // Multi-touch pinch zoom
        this.diagramContainer.addEventListener('touchstart', (e) => {
            if (e.touches.length === 2) {
                // Calculate initial distance between two touch points
                const touch1 = e.touches[0];
                const touch2 = e.touches[1];
                this.initialDistance = Math.hypot(
                    touch2.pageX - touch1.pageX, 
                    touch2.pageY - touch1.pageY
                );
                e.preventDefault();
            }
        });

        this.diagramContainer.addEventListener('touchmove', (e) => {
            if (e.touches.length === 2) {
                const touch1 = e.touches[0];
                const touch2 = e.touches[1];
                
                // Calculate current distance
                this.currentDistance = Math.hypot(
                    touch2.pageX - touch1.pageX, 
                    touch2.pageY - touch1.pageY
                );
                
                // Calculate zoom change
                const zoomFactor = this.currentDistance / this.initialDistance;
                
                // Update zoom level
                this.zoomLevel *= zoomFactor;
                this.zoomLevel = Math.max(0.5, Math.min(3, this.zoomLevel));
                
                // Update initial distance for next move
                this.initialDistance = this.currentDistance;
                
                // Apply zoom
                this.updateZoom();
                
                e.preventDefault();
            }
        });

        // Prevent default touch behaviors that might interfere
        this.diagramContainer.addEventListener('touchend', (e) => {
            this.initialDistance = 0;
        });
    }

    setupMouseZoom() {
        // Mouse wheel zoom
        this.diagramContainer.addEventListener('wheel', (e) => {
            // Prevent default scrolling
            e.preventDefault();
            
            // Determine zoom direction
            const delta = e.deltaY > 0 ? 0.9 : 1.1;
            
            // Update zoom level
            this.zoomLevel *= delta;
            this.zoomLevel = Math.max(0.5, Math.min(3, this.zoomLevel));
            
            // Apply zoom
            this.updateZoom();
        }, { passive: false });
    }

    createIconButton(text, onClick, title) {
        const btn = document.createElement('button');
        btn.textContent = text;
        btn.title = title;
        btn.className = 'w-8 h-8 flex items-center justify-center rounded-md hover:bg-gray-100 dark:hover:bg-gray-700 transition-colors';
        btn.addEventListener('click', onClick);
        return btn;
    }

    open(mermaidCode) {
        // Clear previous content
        this.diagramContainer.innerHTML = '';
        
        // Create mermaid div
        const mermaidDiv = document.createElement('div');
        mermaidDiv.className = 'mermaid w-full';
        mermaidDiv.textContent = mermaidCode;
        
        this.diagramContainer.appendChild(mermaidDiv);
        
        // Initialize mermaid
        mermaid.initialize({ 
            startOnLoad: true,
            theme: document.documentElement.classList.contains('dark') ? 'dark' : 'default'
        });
        mermaid.init();
        
        // Show modal
        this.modal.style.display = 'flex';
        
        // Reset zoom and position
        this.resetZoom();
    }

    close() {
        this.modal.style.display = 'none';
    }

    zoomIn() {
        this.zoomLevel = Math.min(this.zoomLevel * 1.2, 3);
        this.updateZoom();
    }

    zoomOut() {
        this.zoomLevel = Math.max(this.zoomLevel / 1.2, 0.5);
        this.updateZoom();
    }

    resetZoom() {
        this.zoomLevel = 1;
        this.xOffset = 0;
        this.yOffset = 0;
        this.updateZoom();
    }

    updateZoom() {
        // Apply both zoom and dragging to the diagram container
        this.diagramContainer.style.transform = `translate(${this.xOffset}px, ${this.yOffset}px) scale(${this.zoomLevel})`;
        this.diagramContainer.style.transformOrigin = 'center center';
    }

    downloadDiagram() {
        html2canvas(this.diagramContainer).then((canvas) => {
            const imgData = canvas.toDataURL('image/png');
            const link = document.createElement('a');
            link.href = imgData;
            link.download = 'diagram.png';
            link.click();
        });
    }
}
