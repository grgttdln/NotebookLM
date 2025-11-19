// API Base URL
const API_BASE = "http://localhost:8000/api";

// State
let documents = [];
let selectedDocumentId = null;
let conversationId = null;
let selectedDocumentIds = new Set(); // Track which documents are selected for queries

// Initialize
document.addEventListener("DOMContentLoaded", () => {
  setupEventListeners();
  setupResizers();
  loadPanelSizes();
  loadDocuments();
});

function setupEventListeners() {
  // Upload button
  document.getElementById("uploadBtn").addEventListener("click", () => {
    document.getElementById("fileInput").click();
  });

  // File input
  document
    .getElementById("fileInput")
    .addEventListener("change", handleFileUpload);

  // Send button
  document.getElementById("sendBtn").addEventListener("click", sendMessage);

  // Enter key in chat input
  const chatInput = document.getElementById("chatInput");
  chatInput.addEventListener("keydown", (e) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      sendMessage();
    }
  });

  // Auto-resize textarea
  chatInput.addEventListener("input", function () {
    this.style.height = "auto";
    this.style.height = Math.min(this.scrollHeight, 200) + "px";
  });

  // Document selector
  document
    .getElementById("documentSelector")
    .addEventListener("change", (e) => {
      loadDocument(e.target.value);
    });

  // Select all checkbox
  document
    .getElementById("selectAllCheckbox")
    .addEventListener("change", (e) => {
      toggleSelectAll(e.target.checked);
    });
}

async function loadDocuments() {
  try {
    const response = await fetch(`${API_BASE}/documents`);
    const data = await response.json();
    const newDocuments = data.documents || [];

    // Preserve selection state for existing documents, select new ones by default
    const existingIds = new Set(documents.map((d) => d.id));
    const newIds = new Set(newDocuments.map((d) => d.id));

    // Remove selections for deleted documents
    selectedDocumentIds.forEach((id) => {
      if (!newIds.has(id)) {
        selectedDocumentIds.delete(id);
      }
    });

    // Select all new documents by default
    newDocuments.forEach((doc) => {
      if (!existingIds.has(doc.id)) {
        selectedDocumentIds.add(doc.id);
      }
    });

    documents = newDocuments;
    renderDocumentList();
    updateDocumentSelector();
  } catch (error) {
    console.error("Error loading documents:", error);
  }
}

function renderDocumentList() {
  const list = document.getElementById("documentList");
  const header = document.getElementById("documentListHeader");

  if (documents.length === 0) {
    list.innerHTML = `
      <div class="empty-state">
        <span class="material-icons empty-state-icon" aria-hidden="true">description</span>
        <p>No documents uploaded yet</p>
        <p class="empty-state-hint">Upload documents to get started</p>
      </div>
    `;
    header.style.display = "none";
    return;
  }

  header.style.display = "block";

  // Update select all checkbox state
  const selectAllCheckbox = document.getElementById("selectAllCheckbox");
  const allSelected =
    documents.length > 0 &&
    documents.every((doc) => selectedDocumentIds.has(doc.id));
  selectAllCheckbox.checked = allSelected;
  selectAllCheckbox.indeterminate =
    !allSelected && documents.some((doc) => selectedDocumentIds.has(doc.id));

  list.innerHTML = documents
    .map((doc) => {
      const isSelected = selectedDocumentIds.has(doc.id);
      let fileType = doc.file_type || "PDF";
      // Remove leading dot if present
      if (fileType.startsWith(".")) {
        fileType = fileType.substring(1);
      }
      const fileTypeClass = fileType.toLowerCase();

      return `
        <div class="document-item ${
          doc.id === selectedDocumentId ? "active" : ""
        }">
            <div class="document-item-content" onclick="selectDocument('${
              doc.id
            }')">
                <div class="document-item-left">
                    <label class="document-checkbox-label" onclick="event.stopPropagation()">
                        <input 
                            type="checkbox" 
                            class="document-checkbox" 
                            data-doc-id="${doc.id}"
                            ${isSelected ? "checked" : ""}
                            onchange="toggleDocumentSelection('${
                              doc.id
                            }', this.checked)">
                    </label>
                    <div class="file-type-icon ${fileTypeClass}">${fileType}</div>
                    <div class="document-item-info">
                        <div class="document-item-name">${escapeHtml(
                          doc.file_name
                        )}</div>
                        <div class="document-item-meta">
                            ${formatFileSize(doc.text_length)} • ${
        doc.word_count
      } words
                        </div>
                    </div>
                </div>
                <div class="document-item-actions">
                    <button class="document-menu-btn" onclick="event.stopPropagation(); showDocumentMenu(event, '${
                      doc.id
                    }')" aria-label="Document options">
                        <span class="material-icons" aria-hidden="true">more_vert</span>
                    </button>
                </div>
            </div>
        </div>
    `;
    })
    .join("");
}

function updateDocumentSelector() {
  const selector = document.getElementById("documentSelector");

  if (documents.length === 0) {
    selector.style.display = "none";
    return;
  }

  selector.style.display = "block";
  selector.innerHTML =
    '<option value="">Select a document...</option>' +
    documents
      .map(
        (doc) =>
          `<option value="${doc.id}">${escapeHtml(doc.file_name)}</option>`
      )
      .join("");
}

async function handleFileUpload(event) {
  const files = event.target.files;
  if (!files || files.length === 0) return;

  for (const file of files) {
    await uploadFile(file);
  }

  // Reset input
  event.target.value = "";
}

async function uploadFile(file) {
  const modal = document.getElementById("uploadModal");
  const progressBar = document.getElementById("progressBar");
  const progressText = document.getElementById("progressText");

  modal.style.display = "flex";
  progressBar.style.width = "0%";
  progressBar.setAttribute("aria-valuenow", "0");
  progressText.textContent = `Uploading ${file.name}...`;

  try {
    const formData = new FormData();
    formData.append("file", file);

    progressBar.style.width = "30%";
    progressBar.setAttribute("aria-valuenow", "30");
    progressText.textContent = "Processing document...";

    const response = await fetch(`${API_BASE}/upload`, {
      method: "POST",
      body: formData,
    });

    if (!response.ok) {
      const error = await response.json();
      throw new Error(error.detail || "Upload failed");
    }

    progressBar.style.width = "70%";
    progressBar.setAttribute("aria-valuenow", "70");
    progressText.textContent = "Indexing document...";

    const result = await response.json();

    progressBar.style.width = "100%";
    progressBar.setAttribute("aria-valuenow", "100");
    progressText.textContent = "Complete!";

    setTimeout(() => {
      modal.style.display = "none";
      loadDocuments();
    }, 500);
  } catch (error) {
    modal.style.display = "none";
    alert(`Upload failed: ${error.message}`);
    console.error("Upload error:", error);
  }
}

async function sendMessage() {
  const input = document.getElementById("chatInput");
  const question = input.value.trim();

  if (!question) return;

  // Add user message to chat
  addMessage("user", question);
  input.value = "";

  // Disable send button
  const sendBtn = document.getElementById("sendBtn");
  sendBtn.disabled = true;
  sendBtn.innerHTML = '<span class="loading"></span>';

  try {
    // Get selected document IDs (if none selected, use all documents)
    const docIds =
      selectedDocumentIds.size > 0 ? Array.from(selectedDocumentIds) : null;

    const response = await fetch(`${API_BASE}/chat`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        question: question,
        conversation_id: conversationId,
        document_ids: docIds,
      }),
    });

    if (!response.ok) {
      const error = await response.json();
      throw new Error(error.detail || "Chat request failed");
    }

    const data = await response.json();
    conversationId = data.conversation_id;

    // Add assistant message with citations
    addMessage("assistant", data.answer, data.citations);
  } catch (error) {
    addMessage("assistant", `Error: ${error.message}`, []);
  } finally {
    sendBtn.disabled = false;
    sendBtn.innerHTML =
      '<span class="material-icons" aria-hidden="true">send</span>';
  }
}

function addMessage(role, content, citations = []) {
  const container = document.getElementById("chatContainer");

  // Remove welcome message if present
  const welcomeMsg = container.querySelector(".welcome-message");
  if (welcomeMsg) {
    welcomeMsg.remove();
  }

  const messageDiv = document.createElement("div");
  messageDiv.className = `message ${role}`;

  const contentDiv = document.createElement("div");
  contentDiv.className = "message-content";

  if (role === "assistant") {
    // Render Markdown content
    if (typeof marked !== "undefined") {
      // Configure marked options (using use() for v11+)
      marked.use({
        breaks: true,
        gfm: true,
      });

      // Render Markdown to HTML
      const htmlContent = marked.parse(content);
      contentDiv.innerHTML = htmlContent;
    } else {
      // Fallback to plain text if marked is not available
      const paragraphs = content.split("\n\n");
      paragraphs.forEach((para) => {
        const p = document.createElement("p");
        p.textContent = para;
        contentDiv.appendChild(p);
      });
    }

    // Add citations
    if (citations && citations.length > 0) {
      const citationsDiv = document.createElement("div");
      citationsDiv.className = "citations";

      const title = document.createElement("div");
      title.className = "citations-title";
      title.textContent = "Sources:";
      citationsDiv.appendChild(title);

      citations.forEach((citation) => {
        const citationLink = document.createElement("a");
        citationLink.className = "citation";
        citationLink.textContent = `[${citation.citation_id}] ${citation.file_name}`;
        citationLink.href = "#";
        citationLink.onclick = (e) => {
          e.preventDefault();
          showCitation(citation);
        };
        citationsDiv.appendChild(citationLink);
      });

      contentDiv.appendChild(citationsDiv);
    }
  } else {
    contentDiv.textContent = content;
  }

  messageDiv.appendChild(contentDiv);
  container.appendChild(messageDiv);
  container.scrollTop = container.scrollHeight;
}

function showCitation(citation) {
  // Load the document and highlight the chunk
  loadDocument(citation.document_id, citation.chunk_index);

  // Also show the chunk text in a highlighted way
  const viewer = document.getElementById("documentViewer");
  setTimeout(() => {
    // Try to find and highlight the chunk text
    const content = viewer.querySelector(".document-content");
    if (content && citation.chunk_text) {
      // Simple text highlighting (basic implementation)
      const text = content.textContent;
      const chunkPreview = citation.chunk_text.substring(0, 100);
      const index = text.indexOf(chunkPreview);
      if (index !== -1) {
        // Scroll to the found position
        const lines = text.substring(0, index).split("\n").length;
        viewer.scrollTop = lines * 20; // Approximate line height
      }
    }
  }, 200);
}

async function selectDocument(documentId) {
  selectedDocumentId = documentId;
  renderDocumentList();
  await loadDocument(documentId);
}

async function loadDocument(documentId, highlightChunkIndex = null) {
  if (!documentId) {
    document.getElementById("documentViewer").innerHTML =
      '<div class="empty-state"><span class="material-icons empty-state-icon" aria-hidden="true">visibility</span><p>Select a document or click on a citation to view content</p></div>';
    return;
  }

  try {
    const response = await fetch(`${API_BASE}/documents/${documentId}`);
    if (!response.ok) {
      throw new Error("Document not found");
    }

    const data = await response.json();

    // Ensure document_id is in metadata
    if (!data.metadata.document_id) {
      data.metadata.document_id = documentId;
    }

    // Debug logging
    console.log("Document data:", {
      document_id: documentId,
      file_url: data.file_url,
      file_type: data.file_type,
      file_name: data.metadata?.file_name,
    });

    displayDocument(
      data.text,
      data.metadata,
      data.file_url,
      data.file_type,
      highlightChunkIndex
    );

    // Update selector
    document.getElementById("documentSelector").value = documentId;
    selectedDocumentId = documentId;
    renderDocumentList();
  } catch (error) {
    document.getElementById(
      "documentViewer"
    ).innerHTML = `<div class="empty-state"><span class="material-icons empty-state-icon" aria-hidden="true">error_outline</span><p>Error loading document: ${error.message}</p></div>`;
  }
}

function displayDocument(
  text,
  metadata,
  fileUrl,
  fileType,
  highlightChunkIndex = null
) {
  const viewer = document.getElementById("documentViewer");

  // Display document metadata
  const metadataDiv = document.createElement("div");
  metadataDiv.className = "document-metadata";
  metadataDiv.innerHTML = `
        <div style="display: flex; align-items: center; gap: 12px; margin-bottom: 8px;">
          <span class="material-icons" style="font-size: 18px; color: var(--md-on-surface); opacity: 0.4;">description</span>
          <strong style="font-size: 14px; font-weight: 400; color: var(--md-on-surface);">${escapeHtml(
            metadata.file_name || "Document"
          )}</strong>
        </div>
        <div style="font-size: 12px; color: var(--md-on-surface-variant); opacity: 0.6;">
          ${metadata.word_count ? `${metadata.word_count} words` : ""}
          ${metadata.page_count ? ` • ${metadata.page_count} pages` : ""}
        </div>
    `;

  viewer.innerHTML = "";
  viewer.appendChild(metadataDiv);

  // Determine file type - check fileType parameter or extract from file_name
  let actualFileType = fileType;
  if (!actualFileType && metadata.file_name) {
    const fileName = metadata.file_name.toLowerCase();
    if (fileName.endsWith(".pdf")) actualFileType = ".pdf";
    else if (fileName.endsWith(".docx")) actualFileType = ".docx";
    else if (fileName.endsWith(".doc")) actualFileType = ".doc";
  }

  // Normalize file type (remove leading dot if present for comparison)
  const normalizedFileType = actualFileType
    ? actualFileType.toLowerCase().replace(/^\./, "")
    : "";

  // If fileUrl is not provided but we have a PDF, try to construct it
  let finalFileUrl = fileUrl;
  if (!finalFileUrl && normalizedFileType === "pdf" && metadata.file_name) {
    // Try to construct file URL from document ID (if available in metadata)
    const docId = metadata.document_id;
    if (docId) {
      finalFileUrl = `/api/documents/${docId}/file`;
    }
  }

  // If we have a file URL and it's a PDF, display it using an iframe
  if (
    finalFileUrl &&
    (normalizedFileType === "pdf" || actualFileType === ".pdf")
  ) {
    console.log("Displaying PDF:", finalFileUrl);
    const pdfContainer = document.createElement("div");
    pdfContainer.className = "pdf-viewer-container";

    const pdfIframe = document.createElement("iframe");
    pdfIframe.src = finalFileUrl;
    pdfIframe.title = escapeHtml(metadata.file_name || "Document");

    pdfContainer.appendChild(pdfIframe);
    viewer.appendChild(pdfContainer);
  } else if (
    finalFileUrl &&
    (normalizedFileType === "docx" ||
      normalizedFileType === "doc" ||
      actualFileType === ".docx" ||
      actualFileType === ".doc")
  ) {
    // For Word documents, show text content inline with optional download button
    const actionsDiv = document.createElement("div");
    actionsDiv.className = "document-actions";
    actionsDiv.style.cssText =
      "margin-bottom: 24px; padding: 16px 0; display: flex; justify-content: space-between; align-items: center; gap: 12px;";

    const downloadBtn = document.createElement("button");
    downloadBtn.className = "mdc-button mdc-button--raised";
    downloadBtn.innerHTML =
      '<span class="material-icons" aria-hidden="true">download</span><span>Download</span>';
    downloadBtn.onclick = (e) => {
      e.preventDefault();
      const link = document.createElement("a");
      link.href = finalFileUrl;
      link.download = escapeHtml(metadata.file_name || "document");
      document.body.appendChild(link);
      link.click();
      document.body.removeChild(link);
    };

    actionsDiv.innerHTML = `<span style="font-size: 14px; color: var(--md-on-surface); font-weight: 400; display: flex; align-items: center; gap: 8px; opacity: 0.8;"><span class="material-icons" style="font-size: 18px; opacity: 0.4;">description</span>${escapeHtml(
      metadata.file_name || "Document"
    )}</span>`;
    actionsDiv.appendChild(downloadBtn);
    viewer.appendChild(actionsDiv);

    // Show text content inline
    const contentDiv = document.createElement("div");
    contentDiv.className = "document-content";
    contentDiv.textContent = text;
    viewer.appendChild(contentDiv);
  } else {
    // For text files or when file URL is not available, show text content
    console.log("Displaying text content (no file URL or not PDF/Word)");
    const contentDiv = document.createElement("div");
    contentDiv.className = "document-content";

    if (
      highlightChunkIndex !== null &&
      typeof highlightChunkIndex === "number"
    ) {
      // Try to highlight the specific chunk
      contentDiv.textContent = text;
      viewer.appendChild(contentDiv);

      // Scroll to approximate position
      setTimeout(() => {
        const chunkSize = 1000; // Approximate chunk size
        viewer.scrollTop = highlightChunkIndex * chunkSize;
      }, 100);
    } else {
      contentDiv.textContent = text;
      viewer.appendChild(contentDiv);
    }
  }
}

// Utility functions
function escapeHtml(text) {
  const div = document.createElement("div");
  div.textContent = text;
  return div.innerHTML;
}

function formatFileSize(bytes) {
  if (bytes < 1024) return bytes + " B";
  if (bytes < 1024 * 1024) return (bytes / 1024).toFixed(1) + " KB";
  return (bytes / (1024 * 1024)).toFixed(1) + " MB";
}

// Document selection functions
function toggleDocumentSelection(docId, checked) {
  if (checked) {
    selectedDocumentIds.add(docId);
  } else {
    selectedDocumentIds.delete(docId);
  }
  updateSelectAllCheckbox();
}

function toggleSelectAll(checked) {
  if (checked) {
    documents.forEach((doc) => selectedDocumentIds.add(doc.id));
  } else {
    selectedDocumentIds.clear();
  }
  renderDocumentList();
}

function updateSelectAllCheckbox() {
  const selectAllCheckbox = document.getElementById("selectAllCheckbox");
  if (!selectAllCheckbox) return;

  const allSelected =
    documents.length > 0 &&
    documents.every((doc) => selectedDocumentIds.has(doc.id));
  const someSelected = documents.some((doc) => selectedDocumentIds.has(doc.id));

  selectAllCheckbox.checked = allSelected;
  selectAllCheckbox.indeterminate = !allSelected && someSelected;
}

// Document menu functions
let currentMenuDocId = null;

function showDocumentMenu(event, docId) {
  // Close any existing menu
  closeDocumentMenu();

  event.stopPropagation();
  currentMenuDocId = docId;

  const menu = document.createElement("div");
  menu.className = "document-context-menu";
  menu.id = "documentContextMenu";
  menu.innerHTML = `
    <div class="menu-item" onclick="removeDocument('${docId}')">
      <span class="material-icons" aria-hidden="true">delete_outline</span>
      <span>Remove source</span>
    </div>
  `;

  document.body.appendChild(menu);

  // Position menu near the button
  const buttonRect = event.currentTarget.getBoundingClientRect();
  const menuRect = menu.getBoundingClientRect();

  let left = buttonRect.right - menuRect.width;
  let top = buttonRect.bottom + 4;

  // Adjust if menu goes off screen
  if (left < 0) {
    left = buttonRect.left;
  }
  if (top + menuRect.height > window.innerHeight) {
    top = buttonRect.top - menuRect.height - 4;
  }

  menu.style.left = `${left}px`;
  menu.style.top = `${top}px`;

  // Close menu when clicking outside
  setTimeout(() => {
    document.addEventListener("click", closeDocumentMenu, { once: true });
  }, 0);
}

function closeDocumentMenu() {
  const menu = document.getElementById("documentContextMenu");
  if (menu) {
    menu.remove();
  }
  currentMenuDocId = null;
}

async function removeDocument(docId) {
  if (!confirm("Are you sure you want to remove this document?")) {
    closeDocumentMenu();
    return;
  }

  try {
    const response = await fetch(`${API_BASE}/documents/${docId}`, {
      method: "DELETE",
    });

    if (!response.ok) {
      const error = await response.json();
      throw new Error(error.detail || "Failed to delete document");
    }

    // Remove from selected set
    selectedDocumentIds.delete(docId);

    // Reload documents
    await loadDocuments();

    // Clear document viewer if deleted document was selected
    if (selectedDocumentId === docId) {
      selectedDocumentId = null;
      document.getElementById("documentViewer").innerHTML =
        '<div class="empty-state"><span class="material-icons empty-state-icon" aria-hidden="true">visibility</span><p>Select a document or click on a citation to view content</p></div>';
    }

    closeDocumentMenu();
  } catch (error) {
    alert(`Error deleting document: ${error.message}`);
    console.error("Delete error:", error);
  }
}

// Resizer functionality
function setupResizers() {
  const leftPane = document.getElementById("leftPane");
  const centerPane = document.getElementById("centerPane");
  const rightPane = document.getElementById("rightPane");
  const leftResizer = document.getElementById("leftResizer");
  const rightResizer = document.getElementById("rightResizer");

  if (!leftPane || !centerPane || !rightPane || !leftResizer || !rightResizer) {
    return;
  }

  let isResizing = false;
  let currentResizer = null;
  let startX = 0;
  let startLeftWidth = 0;
  let startRightWidth = 0;
  let rafId = null;

  function startResize(e, resizer) {
    if (window.innerWidth <= 768) return; // Disable on mobile

    isResizing = true;
    currentResizer = resizer;
    startX = e.clientX || (e.touches && e.touches[0].clientX);

    const leftRect = leftPane.getBoundingClientRect();
    const rightRect = rightPane.getBoundingClientRect();

    startLeftWidth = leftRect.width;
    startRightWidth = rightRect.width;

    document.body.style.cursor = "col-resize";
    document.body.style.userSelect = "none";

    e.preventDefault();
  }

  function doResize(e) {
    if (!isResizing) return;

    if (rafId) {
      cancelAnimationFrame(rafId);
    }

    rafId = requestAnimationFrame(() => {
      const currentX = e.clientX || (e.touches && e.touches[0].clientX);
      const diff = currentX - startX;
      const containerWidth =
        document.querySelector(".app-container").offsetWidth;

      if (currentResizer === leftResizer) {
        const newLeftWidth = startLeftWidth + diff;
        const minLeftWidth = 200;
        const maxLeftWidth = containerWidth * 0.5;

        if (newLeftWidth >= minLeftWidth && newLeftWidth <= maxLeftWidth) {
          leftPane.style.width = `${newLeftWidth}px`;
        }
      } else if (currentResizer === rightResizer) {
        const newRightWidth = startRightWidth - diff;
        const minRightWidth = 250;
        const maxRightWidth = containerWidth * 0.5;

        if (newRightWidth >= minRightWidth && newRightWidth <= maxRightWidth) {
          rightPane.style.width = `${newRightWidth}px`;
        }
      }
    });

    e.preventDefault();
  }

  function stopResize() {
    if (!isResizing) return;

    if (rafId) {
      cancelAnimationFrame(rafId);
      rafId = null;
    }

    isResizing = false;
    currentResizer = null;
    document.body.style.cursor = "";
    document.body.style.userSelect = "";

    // Save panel sizes after resize completes
    savePanelSizes();
  }

  // Mouse events
  leftResizer.addEventListener("mousedown", (e) => startResize(e, leftResizer));
  rightResizer.addEventListener("mousedown", (e) =>
    startResize(e, rightResizer)
  );

  document.addEventListener("mousemove", doResize);
  document.addEventListener("mouseup", stopResize);

  // Touch events for mobile
  leftResizer.addEventListener(
    "touchstart",
    (e) => startResize(e, leftResizer),
    { passive: false }
  );
  rightResizer.addEventListener(
    "touchstart",
    (e) => startResize(e, rightResizer),
    { passive: false }
  );

  document.addEventListener("touchmove", doResize, { passive: false });
  document.addEventListener("touchend", stopResize);

  // Handle window resize
  let resizeTimeout;
  window.addEventListener("resize", () => {
    clearTimeout(resizeTimeout);
    resizeTimeout = setTimeout(() => {
      if (window.innerWidth > 768) {
        loadPanelSizes();
      }
    }, 250);
  });
}

// Panel size persistence
function savePanelSizes() {
  if (window.innerWidth <= 768) return; // Don't save on mobile

  try {
    const leftPane = document.getElementById("leftPane");
    const rightPane = document.getElementById("rightPane");

    if (leftPane && rightPane) {
      const sizes = {
        leftWidth: leftPane.offsetWidth,
        rightWidth: rightPane.offsetWidth,
      };
      localStorage.setItem("notebooklm_panel_sizes", JSON.stringify(sizes));
    }
  } catch (error) {
    console.error("Error saving panel sizes:", error);
  }
}

function loadPanelSizes() {
  if (window.innerWidth <= 768) return; // Don't load on mobile

  try {
    const saved = localStorage.getItem("notebooklm_panel_sizes");
    if (saved) {
      const sizes = JSON.parse(saved);
      const leftPane = document.getElementById("leftPane");
      const rightPane = document.getElementById("rightPane");
      const containerWidth =
        document.querySelector(".app-container").offsetWidth;

      if (leftPane && sizes.leftWidth) {
        const minLeftWidth = 200;
        const maxLeftWidth = containerWidth * 0.5;
        const leftWidth = Math.max(
          minLeftWidth,
          Math.min(maxLeftWidth, sizes.leftWidth)
        );
        leftPane.style.width = `${leftWidth}px`;
      }

      if (rightPane && sizes.rightWidth) {
        const minRightWidth = 250;
        const maxRightWidth = containerWidth * 0.5;
        const rightWidth = Math.max(
          minRightWidth,
          Math.min(maxRightWidth, sizes.rightWidth)
        );
        rightPane.style.width = `${rightWidth}px`;
      }
    }
  } catch (error) {
    console.error("Error loading panel sizes:", error);
  }
}
