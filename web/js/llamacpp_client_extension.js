import { app } from "../../scripts/app.js";

app.registerExtension({
    name: "LlamaCppClient.Extension",
    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        if (nodeData.name === "LlamaCppClient") {
            const onNodeCreated = nodeType.prototype.onNodeCreated;
            
            nodeType.prototype.onNodeCreated = function () {
                if (onNodeCreated) {
                    onNodeCreated.apply(this, arguments);
                }

                const self = this;
                
                // Helper to hide/show widgets
                function toggleWidget(widgetName, show) {
                    const widget = self.widgets?.find(w => w.name === widgetName);
                    if (widget) {
                        widget.type = show ? (widget.origType || widget.type) : "hidden";
                        widget.computeSize = show ? (widget.origComputeSize || widget.computeSize) : () => [0, -4];
                        if (show && widget.origType === "hidden") {
                            widget.type = "text"; // Fallback to text if origType isn't saved properly
                        }
                    }
                }

                // Initial save of original widget properties
                for (const w of this.widgets || []) {
                    w.origType = w.type;
                    w.origComputeSize = w.computeSize;
                }

                // Update UI based on endpoint
                const updateUI = () => {
                    const endpointWidget = this.widgets.find(w => w.name === "endpoint");
                    if (!endpointWidget) return;
                    
                    const isChat = endpointWidget.value === "chat_completions";
                    
                    // Fields only for chat_completions
                    const chatFields = ["messages", "system_message", "user_message", "assistant_message", "max_tokens", "tools", "tool_choice", "response_format"];
                    // Fields only for completion
                    const completionFields = ["prompt", "n_predict"];

                    for (const f of chatFields) {
                        toggleWidget(f, isChat);
                    }
                    for (const f of completionFields) {
                        toggleWidget(f, !isChat);
                    }
                    
                    // Resize node to fit the visible widgets
                    this.setSize(this.computeSize());
                };

                // Add callback to endpoint widget
                const endpointWidget = this.widgets.find(w => w.name === "endpoint");
                if (endpointWidget) {
                    const origCallback = endpointWidget.callback;
                    endpointWidget.callback = function() {
                        if (origCallback) origCallback.apply(this, arguments);
                        updateUI();
                    };
                }

                // Run once on load
                requestAnimationFrame(updateUI);
                
                // Handle connections (hide image_data if images pin is connected)
                const onConnectionsChange = this.onConnectionsChange;
                this.onConnectionsChange = function (type, index, connected, link_info) {
                    if (onConnectionsChange) {
                        onConnectionsChange.apply(this, arguments);
                    }

                    // Check if 'images' input is connected
                    const imagesInputIndex = this.inputs?.findIndex(i => i.name === "images");
                    if (imagesInputIndex !== -1 && imagesInputIndex === index) {
                        toggleWidget("image_data", !connected);
                        this.setSize(this.computeSize());
                    }
                };
                
                // Initial check for connections
                requestAnimationFrame(() => {
                    const imagesInput = this.inputs?.find(i => i.name === "images");
                    if (imagesInput && imagesInput.link !== null && imagesInput.link !== undefined) {
                         toggleWidget("image_data", false);
                    }
                });
            };
        }
    }
});
