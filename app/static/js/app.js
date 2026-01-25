// State
let allData = [];
let jsonPath = '';
let displayedCount = 0;
const BATCH_SIZE = 100;
let isLoading = false;

// Filter State
let hideDeleted = false;
let hideApproved = false;

// File Browser State
let browserMode = 'folder';
let browserTargetInput = null;
let browserCurrentPath = '';
let browserParentPath = null;
let browserSelectedItem = null;

// Image Popup State
let popupZoom = 100;
let popupMode = 'normal'; // 'normal', 'overlay', 'toggle'
let popupImage1Src = '';
let popupImage2Src = '';
let popupShowingFirst = true;

// Folder info for apply
let folderInfo = null;

// DOM Elements
const setupPanel = document.getElementById('setup-panel');
const imageContainer = document.getElementById('image-container');
const labelFileInput = document.getElementById('labelFile');
const loadLabelBtn = document.getElementById('load-label-btn');
const folder1Input = document.getElementById('folder1');
const folder2Input = document.getElementById('folder2');
const folder3Input = document.getElementById('folder3');
const startBtn = document.getElementById('start-btn');
const backBtn = document.getElementById('back-btn');
const imageGrid = document.getElementById('image-grid');
const loadingIndicator = document.getElementById('loading-indicator');
const jsonFilenameSpan = document.getElementById('json-filename');
const itemCountSpan = document.getElementById('item-count');
const headerInfo = document.getElementById('header-info');
const statusCountSpan = document.getElementById('status-count');
const hideDeletedCheckbox = document.getElementById('hide-deleted');
const hideApprovedCheckbox = document.getElementById('hide-approved');
const applyBtn = document.getElementById('apply-btn');

// Modal Elements
const browserModal = document.getElementById('browser-modal');
const modalTitle = document.getElementById('modal-title');
const modalClose = document.getElementById('modal-close');
const modalUpBtn = document.getElementById('modal-up-btn');
const modalPathInput = document.getElementById('modal-path-input');
const modalGoBtn = document.getElementById('modal-go-btn');
const modalFileList = document.getElementById('modal-file-list');
const modalSelected = document.getElementById('modal-selected');
const modalCancel = document.getElementById('modal-cancel');
const modalSelect = document.getElementById('modal-select');

// Image Popup Elements
const imagePopup = document.getElementById('image-popup');
const popupImage = document.getElementById('popup-image');
const popupImage2 = document.getElementById('popup-image2');
const popupTitle = document.getElementById('popup-title');
const popupZoomLevel = document.getElementById('popup-zoom-level');
const popupClose = document.getElementById('popup-close');
const popupToggleBtn = document.getElementById('popup-toggle-btn');
const popupImageContainer = document.querySelector('.popup-image-container');

// Apply Modal Elements
const applyModal = document.getElementById('apply-modal');
const applyModalClose = document.getElementById('apply-modal-close');
const applyTargetPath = document.getElementById('apply-target-path');
const applyBrowseBtn = document.getElementById('apply-browse-btn');
const applyTargetFolders = document.getElementById('apply-target-folders');
const applyStatus = document.getElementById('apply-status');
const applyCancel = document.getElementById('apply-cancel');
const applyConfirm = document.getElementById('apply-confirm');

// LocalStorage keys
const STORAGE_KEY = 'imageLabelingConfig';

// Load saved config on page load
function loadConfig() {
    const saved = localStorage.getItem(STORAGE_KEY);
    if (saved) {
        try {
            const config = JSON.parse(saved);
            folder1Input.value = config.folder1 || '';
            folder2Input.value = config.folder2 || '';
            folder3Input.value = config.folder3 || '';
            labelFileInput.value = config.labelFile || '';
        } catch (e) {
            console.error('Failed to load config:', e);
        }
    }
}

// Save config to localStorage
function saveConfig() {
    const config = {
        folder1: folder1Input.value,
        folder2: folder2Input.value,
        folder3: folder3Input.value,
        labelFile: labelFileInput.value
    };
    localStorage.setItem(STORAGE_KEY, JSON.stringify(config));
}

// Update status count display
function updateStatusCount() {
    const approvedCount = allData.filter(item => item.status === 'approved').length;
    const deletedCount = allData.filter(item => item.status === 'deleted').length;
    statusCountSpan.textContent = `Approved: ${approvedCount} | Deleted: ${deletedCount}`;
}

// Apply filters to rows
function applyFilters() {
    const rows = imageGrid.querySelectorAll('.image-row');
    rows.forEach(row => {
        const id = row.dataset.id;
        const item = allData.find(d => d.id === id);
        if (!item) return;

        let shouldHide = false;
        if (hideDeleted && item.status === 'deleted') shouldHide = true;
        if (hideApproved && item.status === 'approved') shouldHide = true;

        if (shouldHide) {
            row.classList.add('hidden-row');
        } else {
            row.classList.remove('hidden-row');
        }
    });
}

// ==================== Image Popup Functions ====================

function openImagePopup(mode, image1Src, image2Src, title) {
    popupMode = mode;
    popupImage1Src = image1Src;
    popupImage2Src = image2Src || '';
    popupZoom = 100;
    popupShowingFirst = true;

    popupImage.src = image1Src;
    popupImage.style.transform = 'scale(1)';
    popupImage.classList.remove('zoomed', 'hidden');

    popupTitle.textContent = title;
    popupZoomLevel.textContent = '100%';

    // Reset container
    popupImageContainer.classList.remove('overlay-mode');

    if (mode === 'overlay' && image2Src) {
        popupImage2.src = image2Src;
        popupImage2.classList.remove('hidden');
        popupImage2.style.transform = 'scale(1)';
        popupImageContainer.classList.add('overlay-mode');
        popupToggleBtn.classList.add('hidden');
    } else if (mode === 'toggle' && image2Src) {
        popupImage2.src = image2Src;
        popupImage2.classList.add('hidden');
        popupToggleBtn.classList.remove('hidden');
        popupToggleBtn.textContent = 'Show Output';
    } else {
        popupImage2.classList.add('hidden');
        popupToggleBtn.classList.add('hidden');
    }

    imagePopup.classList.remove('hidden');
}

function closeImagePopup() {
    imagePopup.classList.add('hidden');
    popupImage.src = '';
    popupImage2.src = '';
    popupMode = 'normal';
}

function handlePopupToggle() {
    if (popupMode !== 'toggle') return;

    popupShowingFirst = !popupShowingFirst;

    if (popupShowingFirst) {
        popupImage.classList.remove('hidden');
        popupImage2.classList.add('hidden');
        popupToggleBtn.textContent = 'Show Output';
    } else {
        const imageWidth = popupImage.offsetWidth;
        const imageHeight = popupImage.offsetHeight;
        popupImage.classList.add('hidden');
        popupImage2.classList.remove('hidden');
        popupToggleBtn.textContent = 'Show Input';

        // Match the size of image2 to image1
        if (popupImage.naturalWidth && popupImage.naturalHeight) {
            popupImage2.style.width = imageWidth + 'px';
            popupImage2.style.height = imageHeight + 'px';
            popupImage2.style.objectFit = 'contain';
        }
    }
}

function handlePopupImageClick(e) {
    e.preventDefault();

    if (popupMode === 'overlay') {
        // In overlay mode, zoom both images
        if (e.altKey) {
            popupZoom = Math.max(10, popupZoom - 10);
        } else {
            popupZoom = Math.min(500, popupZoom + 10);
        }
        popupImage.style.transform = `scale(${popupZoom / 100})`;
        popupImage2.style.transform = `scale(${popupZoom / 100})`;
    } else {
        if (e.altKey) {
            popupZoom = Math.max(10, popupZoom - 10);
        } else {
            popupZoom = Math.min(500, popupZoom + 10);
        }
        const activeImg = popupShowingFirst ? popupImage : popupImage2;
        activeImg.style.transform = `scale(${popupZoom / 100})`;
    }

    popupZoomLevel.textContent = `${popupZoom}%`;

    if (popupZoom > 100) {
        popupImage.classList.add('zoomed');
        popupImage2.classList.add('zoomed');
    } else {
        popupImage.classList.remove('zoomed');
        popupImage2.classList.remove('zoomed');
    }
}

// ==================== Apply Modal Functions ====================

function openApplyModal() {
    if (!folderInfo) {
        alert('Folder information not available');
        return;
    }

    const approvedCount = allData.filter(item => item.status === 'approved').length;
    if (approvedCount === 0) {
        alert('No approved items to apply');
        return;
    }

    applyTargetPath.value = '';
    applyStatus.textContent = `${approvedCount} approved items will be copied`;
    updateApplyTargetFolders();
    applyModal.classList.remove('hidden');
}

function closeApplyModal() {
    applyModal.classList.add('hidden');
}

function updateApplyTargetFolders() {
    const targetPath = applyTargetPath.value.trim();
    if (!targetPath || !folderInfo) {
        applyTargetFolders.innerHTML = '<li>Select a target path first</li>';
        return;
    }

    const folder1Name = folderInfo.folder1.split(/[/\\]/).pop();
    const folder2Name = folderInfo.folder2.split(/[/\\]/).pop();
    const folder3Name = folderInfo.folder3.split(/[/\\]/).pop();

    applyTargetFolders.innerHTML = `
        <li>${targetPath}\\${folder1Name}</li>
        <li>${targetPath}\\${folder2Name}</li>
        <li>${targetPath}\\${folder3Name}</li>
    `;
}

async function executeApply() {
    const targetPath = applyTargetPath.value.trim();
    if (!targetPath) {
        alert('Please select a target path');
        return;
    }

    applyConfirm.disabled = true;
    applyStatus.textContent = 'Copying files...';

    try {
        const response = await fetch('/api/apply-approved', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                json_path: jsonPath,
                target_path: targetPath
            })
        });

        const result = await response.json();

        if (result.error) {
            alert(result.error);
            applyStatus.textContent = 'Error: ' + result.error;
        } else {
            applyStatus.textContent = `Successfully copied ${result.copied_count} items`;
            setTimeout(closeApplyModal, 2000);
        }
    } catch (e) {
        console.error('Failed to apply:', e);
        applyStatus.textContent = 'Error: ' + e.message;
    } finally {
        applyConfirm.disabled = false;
    }
}

// ==================== File Browser Functions ====================

function openFileBrowser(targetInputId, mode) {
    browserTargetInput = document.getElementById(targetInputId);
    browserMode = mode;
    browserSelectedItem = null;

    modalTitle.textContent = mode === 'folder' ? 'Select Folder' : 'Select File';
    modalSelected.textContent = 'No selection';
    modalSelect.disabled = true;

    const startPath = browserTargetInput.value || '';
    browserCurrentPath = startPath;
    modalPathInput.value = startPath;

    loadDirectory(startPath);
    browserModal.classList.remove('hidden');
}

function closeFileBrowser() {
    browserModal.classList.add('hidden');
    browserTargetInput = null;
    browserSelectedItem = null;
}

async function loadDirectory(path) {
    modalFileList.innerHTML = '<div class="loading">Loading...</div>';

    try {
        const response = await fetch(`/api/browse?path=${encodeURIComponent(path)}`);
        const result = await response.json();

        if (result.error) {
            modalFileList.innerHTML = `<div class="error">${result.error}</div>`;
            return;
        }

        browserCurrentPath = result.current_path;
        browserParentPath = result.parent_path;
        modalPathInput.value = browserCurrentPath;

        modalUpBtn.disabled = browserParentPath === null;

        renderFileList(result.items);

    } catch (e) {
        modalFileList.innerHTML = `<div class="error">Failed to load directory</div>`;
    }
}

function renderFileList(items) {
    modalFileList.innerHTML = '';

    if (items.length === 0) {
        modalFileList.innerHTML = '<div class="empty">Empty folder</div>';
        return;
    }

    items.forEach(item => {
        if (browserMode === 'folder' && item.type === 'file') {
            return;
        }

        if (browserMode === 'file' && item.type === 'file') {
            if (item.ext !== '.json') {
                return;
            }
        }

        const div = document.createElement('div');
        div.className = `file-item ${item.type}`;
        div.dataset.path = item.path;
        div.dataset.type = item.type;

        const icon = document.createElement('span');
        icon.className = 'file-item-icon';
        if (item.type === 'drive') {
            icon.textContent = '💾';
        } else if (item.type === 'folder') {
            icon.textContent = '📁';
        } else {
            icon.textContent = '📄';
        }

        const name = document.createElement('span');
        name.className = 'file-item-name';
        name.textContent = item.name;

        div.appendChild(icon);
        div.appendChild(name);

        div.addEventListener('dblclick', () => {
            if (item.type === 'folder' || item.type === 'drive') {
                loadDirectory(item.path);
            }
        });

        div.addEventListener('click', () => {
            modalFileList.querySelectorAll('.file-item.selected').forEach(el => {
                el.classList.remove('selected');
            });

            div.classList.add('selected');
            browserSelectedItem = item;

            if (browserMode === 'folder') {
                if (item.type === 'folder' || item.type === 'drive') {
                    modalSelected.textContent = item.path;
                    modalSelect.disabled = false;
                } else {
                    modalSelected.textContent = 'No selection';
                    modalSelect.disabled = true;
                }
            } else {
                if (item.type === 'file') {
                    modalSelected.textContent = item.path;
                    modalSelect.disabled = false;
                } else {
                    modalSelected.textContent = 'No selection (double-click to enter folder)';
                    modalSelect.disabled = true;
                }
            }
        });

        modalFileList.appendChild(div);
    });

    if (browserMode === 'folder' && browserCurrentPath) {
        modalSelected.textContent = `Current: ${browserCurrentPath}`;
        modalSelect.disabled = false;
        browserSelectedItem = { path: browserCurrentPath, type: 'folder' };
    }
}

function selectCurrentItem() {
    if (!browserTargetInput) return;

    let selectedPath = '';

    if (browserMode === 'folder') {
        selectedPath = browserSelectedItem ? browserSelectedItem.path : browserCurrentPath;
    } else {
        if (browserSelectedItem && browserSelectedItem.type === 'file') {
            selectedPath = browserSelectedItem.path;
        }
    }

    if (selectedPath) {
        browserTargetInput.value = selectedPath;

        // Special handling for apply modal
        if (browserTargetInput.id === 'apply-target-path') {
            updateApplyTargetFolders();
        } else {
            saveConfig();
        }

        closeFileBrowser();
    }
}

// ==================== Image Viewer Functions ====================

function getFilenameWithExt(path) {
    return path.split(/[/\\]/).pop();
}

function calculateAspectRatio(width, height) {
    const gcd = (a, b) => b === 0 ? a : gcd(b, a % b);
    const divisor = gcd(width, height);
    return `${width / divisor}:${height / divisor}`;
}

function createImageViewer(title, content, imagePath) {
    const viewer = document.createElement('div');
    viewer.className = 'image-viewer';

    const titleEl = document.createElement('div');
    titleEl.className = 'viewer-title';
    titleEl.textContent = title;
    viewer.appendChild(titleEl);

    const infoEl = document.createElement('div');
    infoEl.className = 'viewer-info';
    infoEl.textContent = 'Loading...';
    viewer.appendChild(infoEl);

    viewer.appendChild(content);

    if (imagePath) {
        const tempImg = new Image();
        tempImg.onload = function () {
            const ratio = calculateAspectRatio(this.naturalWidth, this.naturalHeight);
            infoEl.textContent = `${this.naturalWidth} × ${this.naturalHeight} (${ratio})`;
        };
        tempImg.onerror = function () {
            infoEl.textContent = 'Failed to load';
        };
        tempImg.src = `/api/image?path=${encodeURIComponent(imagePath)}`;
    }

    return viewer;
}

function createImageButtons(imagePath) {
    const container = document.createElement('div');
    container.className = 'image-buttons';

    // Open in new tab button
    const newTabBtn = document.createElement('button');
    newTabBtn.className = 'image-btn';
    newTabBtn.innerHTML = '↗';
    newTabBtn.title = 'Open in new tab';
    newTabBtn.addEventListener('click', (e) => {
        e.stopPropagation();
        window.open(`/api/image?path=${encodeURIComponent(imagePath)}`, '_blank');
    });

    // Open in explorer button
    const explorerBtn = document.createElement('button');
    explorerBtn.className = 'image-btn';
    explorerBtn.innerHTML = '📂';
    explorerBtn.title = 'Open in Explorer';
    explorerBtn.addEventListener('click', async (e) => {
        e.stopPropagation();
        try {
            await fetch('/api/open-in-explorer', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ path: imagePath })
            });
        } catch (err) {
            console.error('Failed to open in explorer:', err);
        }
    });

    container.appendChild(newTabBtn);
    container.appendChild(explorerBtn);

    return container;
}

function createOverlayViewer(image1Path, image2Path) {
    const wrapper = document.createElement('div');
    wrapper.className = 'image-wrapper overlay-wrapper';

    const img1 = document.createElement('img');
    img1.src = `/api/image?path=${encodeURIComponent(image1Path)}`;
    img1.className = 'overlay-img1';
    img1.loading = 'lazy';

    const img2 = document.createElement('img');
    img2.src = `/api/image?path=${encodeURIComponent(image2Path)}`;
    img2.className = 'overlay-img2';
    img2.loading = 'lazy';

    wrapper.appendChild(img2);
    wrapper.appendChild(img1);

    wrapper.addEventListener('click', () => {
        openImagePopup(
            'overlay',
            `/api/image?path=${encodeURIComponent(image1Path)}`,
            `/api/image?path=${encodeURIComponent(image2Path)}`,
            'Overlay (Input + Output)'
        );
    });

    return createImageViewer('Overlay (Input + Output)', wrapper, image1Path);
}

function createToggleViewer(image1Path, image2Path) {
    const wrapper = document.createElement('div');
    wrapper.className = 'image-wrapper toggle-wrapper';

    const img1 = document.createElement('img');
    img1.src = `/api/image?path=${encodeURIComponent(image1Path)}`;
    img1.className = 'toggle-img1';
    img1.loading = 'lazy';

    const img2 = document.createElement('img');
    img2.src = `/api/image?path=${encodeURIComponent(image2Path)}`;
    img2.className = 'toggle-img2 toggle-hidden';
    img2.loading = 'lazy';

    const toggleBtn = document.createElement('button');
    toggleBtn.className = 'toggle-btn';
    toggleBtn.textContent = 'Toggle';

    let showingFirst = true;
    toggleBtn.addEventListener('click', (e) => {
        e.stopPropagation();
        showingFirst = !showingFirst;
        if (showingFirst) {
            img1.classList.remove('toggle-hidden');
            img2.classList.add('toggle-hidden');
        } else {
            img1.classList.add('toggle-hidden');
            img2.classList.remove('toggle-hidden');
        }
    });

    wrapper.appendChild(img1);
    wrapper.appendChild(img2);
    wrapper.appendChild(toggleBtn);

    wrapper.addEventListener('click', (e) => {
        if (e.target === toggleBtn) return;
        openImagePopup(
            'toggle',
            `/api/image?path=${encodeURIComponent(image1Path)}`,
            `/api/image?path=${encodeURIComponent(image2Path)}`,
            'Toggle (Input / Output)'
        );
    });

    return createImageViewer('Toggle (Input / Output)', wrapper, image1Path);
}

function createStandardViewer(title, imagePath) {
    const wrapper = document.createElement('div');
    wrapper.className = 'image-wrapper';

    const img = document.createElement('img');
    img.src = `/api/image?path=${encodeURIComponent(imagePath)}`;
    img.loading = 'lazy';

    wrapper.appendChild(img);
    wrapper.appendChild(createImageButtons(imagePath));

    wrapper.addEventListener('click', (e) => {
        if (e.target.closest('.image-buttons')) return;
        openImagePopup('normal', `/api/image?path=${encodeURIComponent(imagePath)}`, null, title);
    });

    return createImageViewer(title, wrapper, imagePath);
}

function createImageRow(item) {
    const row = document.createElement('div');
    row.className = 'image-row';
    row.dataset.id = item.id;

    // Apply status class
    if (item.status === 'approved') {
        row.classList.add('approved');
    } else if (item.status === 'deleted') {
        row.classList.add('deleted');
    }

    // Status wrapper with radio buttons
    const statusWrapper = document.createElement('div');
    statusWrapper.className = 'status-wrapper';

    const radioName = `status-${item.id}`;

    // None option
    const noneLabel = document.createElement('label');
    noneLabel.className = 'status-option none-option';
    const noneRadio = document.createElement('input');
    noneRadio.type = 'radio';
    noneRadio.name = radioName;
    noneRadio.value = 'none';
    noneRadio.checked = item.status === 'none' || !item.status;
    noneRadio.addEventListener('change', () => handleStatusChange(item.id, 'none', row));
    noneLabel.appendChild(noneRadio);
    noneLabel.appendChild(document.createTextNode('None'));

    // Approve option
    const approveLabel = document.createElement('label');
    approveLabel.className = 'status-option approve-option';
    const approveRadio = document.createElement('input');
    approveRadio.type = 'radio';
    approveRadio.name = radioName;
    approveRadio.value = 'approved';
    approveRadio.checked = item.status === 'approved';
    approveRadio.addEventListener('change', () => handleStatusChange(item.id, 'approved', row));
    approveLabel.appendChild(approveRadio);
    approveLabel.appendChild(document.createTextNode('Approve'));

    // Delete option
    const deleteLabel = document.createElement('label');
    deleteLabel.className = 'status-option delete-option';
    const deleteRadio = document.createElement('input');
    deleteRadio.type = 'radio';
    deleteRadio.name = radioName;
    deleteRadio.value = 'deleted';
    deleteRadio.checked = item.status === 'deleted';
    deleteRadio.addEventListener('change', () => handleStatusChange(item.id, 'deleted', row));
    deleteLabel.appendChild(deleteRadio);
    deleteLabel.appendChild(document.createTextNode('Delete'));

    statusWrapper.appendChild(noneLabel);
    statusWrapper.appendChild(approveLabel);
    statusWrapper.appendChild(deleteLabel);
    row.appendChild(statusWrapper);

    row.appendChild(createOverlayViewer(item.image1, item.image2));
    row.appendChild(createToggleViewer(item.image1, item.image2));
    row.appendChild(createStandardViewer('Input', item.image1));
    row.appendChild(createStandardViewer('Output', item.image2));
    row.appendChild(createStandardViewer('Reference', item.image3));

    // Row ID with filename and copy button only
    const rowId = document.createElement('div');
    rowId.className = 'row-id';

    const filename = getFilenameWithExt(item.image1);
    const rowIdText = document.createElement('span');
    rowIdText.className = 'row-id-text';
    rowIdText.textContent = filename;

    const copyBtn = document.createElement('button');
    copyBtn.className = 'copy-btn';
    copyBtn.innerHTML = '📋';
    copyBtn.title = 'Copy filename';
    copyBtn.addEventListener('click', async () => {
        try {
            await navigator.clipboard.writeText(filename);
            copyBtn.classList.add('copied');
            copyBtn.innerHTML = '✓';
            setTimeout(() => {
                copyBtn.classList.remove('copied');
                copyBtn.innerHTML = '📋';
            }, 1500);
        } catch (e) {
            console.error('Failed to copy:', e);
        }
    });

    rowId.appendChild(rowIdText);
    rowId.appendChild(copyBtn);
    row.appendChild(rowId);

    return row;
}

async function handleStatusChange(id, status, rowElement) {
    const item = allData.find(d => d.id === id);
    if (item) {
        item.status = status;
    }

    // Update row class
    rowElement.classList.remove('approved', 'deleted');
    if (status === 'approved') {
        rowElement.classList.add('approved');
    } else if (status === 'deleted') {
        rowElement.classList.add('deleted');
    }

    updateStatusCount();
    applyFilters();

    try {
        await fetch('/api/update-item', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ json_path: jsonPath, id, status })
        });
    } catch (e) {
        console.error('Failed to update item:', e);
    }
}

function loadMoreImages() {
    if (isLoading || displayedCount >= allData.length) return;

    isLoading = true;
    loadingIndicator.classList.remove('hidden');

    const endIndex = Math.min(displayedCount + BATCH_SIZE, allData.length);

    for (let i = displayedCount; i < endIndex; i++) {
        const row = createImageRow(allData[i]);
        imageGrid.appendChild(row);
    }

    displayedCount = endIndex;
    isLoading = false;
    loadingIndicator.classList.add('hidden');

    headerInfo.textContent = `${displayedCount} / ${allData.length} items`;
    applyFilters();
}

function handleScroll() {
    const scrollTop = imageGrid.scrollTop;
    const scrollHeight = imageGrid.scrollHeight;
    const clientHeight = imageGrid.clientHeight;

    if (scrollTop + clientHeight >= scrollHeight - 500) {
        loadMoreImages();
    }
}

// Convert old format to new format
function normalizeItemStatus(item) {
    if (item.status) return item;

    // Convert old format
    if (item.approved === true) {
        item.status = 'approved';
    } else if (item.deleted === true) {
        item.status = 'deleted';
    } else {
        item.status = 'none';
    }

    delete item.approved;
    delete item.deleted;

    return item;
}

async function loadLabelFile() {
    const labelPath = labelFileInput.value.trim();
    if (!labelPath) {
        alert('Please enter a label file path');
        return;
    }

    loadLabelBtn.disabled = true;
    loadLabelBtn.textContent = 'Loading...';

    try {
        const response = await fetch(`/api/load-label-file?path=${encodeURIComponent(labelPath)}`);
        const result = await response.json();

        if (result.error) {
            alert(result.error);
            loadLabelBtn.disabled = false;
            loadLabelBtn.textContent = 'Load';
            return;
        }

        jsonPath = labelPath;
        allData = result.data.map(normalizeItemStatus);
        displayedCount = 0;

        if (result.folders) {
            folder1Input.value = result.folders.folder1 || '';
            folder2Input.value = result.folders.folder2 || '';
            folder3Input.value = result.folders.folder3 || '';
            folderInfo = result.folders;
        }

        saveConfig();

        const filename = labelPath.split(/[/\\]/).pop();
        jsonFilenameSpan.textContent = `${filename} (loaded)`;
        itemCountSpan.textContent = `${allData.length} items`;

        setupPanel.classList.add('hidden');
        imageContainer.classList.remove('hidden');

        imageGrid.innerHTML = '';
        loadMoreImages();
        updateStatusCount();

    } catch (e) {
        console.error('Failed to load label file:', e);
        alert('Failed to load label file: ' + e.message);
    } finally {
        loadLabelBtn.disabled = false;
        loadLabelBtn.textContent = 'Load';
    }
}

async function startLabeling() {
    saveConfig();

    const folder1 = folder1Input.value.trim();
    const folder2 = folder2Input.value.trim();
    const folder3 = folder3Input.value.trim();

    if (!folder1 || !folder2 || !folder3) {
        alert('Please enter all 3 folder paths');
        return;
    }

    startBtn.disabled = true;
    startBtn.textContent = 'Loading...';

    try {
        const response = await fetch('/api/scan-folders', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ folder1, folder2, folder3 })
        });

        const result = await response.json();

        if (result.error) {
            alert(result.error);
            startBtn.disabled = false;
            startBtn.textContent = 'Start';
            return;
        }

        jsonPath = result.json_path;
        folderInfo = result.folders;

        if (Array.isArray(result.data)) {
            allData = result.data.map(normalizeItemStatus);
        } else if (result.data && result.data.items) {
            allData = result.data.items.map(normalizeItemStatus);
        } else {
            allData = [];
        }

        displayedCount = 0;

        labelFileInput.value = result.json_path;
        saveConfig();

        jsonFilenameSpan.textContent = result.json_filename;
        itemCountSpan.textContent = `${allData.length} items`;

        if (result.loaded_existing) {
            jsonFilenameSpan.textContent += ' (loaded)';
        }

        setupPanel.classList.add('hidden');
        imageContainer.classList.remove('hidden');

        imageGrid.innerHTML = '';
        loadMoreImages();
        updateStatusCount();

    } catch (e) {
        console.error('Failed to start labeling:', e);
        alert('Failed to start labeling: ' + e.message);
    } finally {
        startBtn.disabled = false;
        startBtn.textContent = 'Start';
    }
}

function backToSetup() {
    imageContainer.classList.add('hidden');
    setupPanel.classList.remove('hidden');
    imageGrid.innerHTML = '';
    displayedCount = 0;
}

// ==================== Event Listeners ====================

// Main buttons
startBtn.addEventListener('click', startLabeling);
loadLabelBtn.addEventListener('click', loadLabelFile);
backBtn.addEventListener('click', backToSetup);
imageGrid.addEventListener('scroll', handleScroll);

// Filter checkboxes
hideDeletedCheckbox.addEventListener('change', () => {
    hideDeleted = hideDeletedCheckbox.checked;
    applyFilters();
});

hideApprovedCheckbox.addEventListener('change', () => {
    hideApproved = hideApprovedCheckbox.checked;
    applyFilters();
});

// Apply button
applyBtn.addEventListener('click', openApplyModal);
applyModalClose.addEventListener('click', closeApplyModal);
applyCancel.addEventListener('click', closeApplyModal);
applyConfirm.addEventListener('click', executeApply);

applyBrowseBtn.addEventListener('click', () => {
    openFileBrowser('apply-target-path', 'folder');
});

applyTargetPath.addEventListener('input', updateApplyTargetFolders);

// Browse buttons
document.querySelectorAll('.browse-btn').forEach(btn => {
    btn.addEventListener('click', () => {
        const target = btn.dataset.target;
        const mode = btn.dataset.mode;
        openFileBrowser(target, mode);
    });
});

// Modal events
modalClose.addEventListener('click', closeFileBrowser);
modalCancel.addEventListener('click', closeFileBrowser);
modalSelect.addEventListener('click', selectCurrentItem);

modalUpBtn.addEventListener('click', () => {
    if (browserParentPath !== null) {
        loadDirectory(browserParentPath);
    }
});

modalGoBtn.addEventListener('click', () => {
    const path = modalPathInput.value.trim();
    loadDirectory(path);
});

modalPathInput.addEventListener('keypress', (e) => {
    if (e.key === 'Enter') {
        const path = modalPathInput.value.trim();
        loadDirectory(path);
    }
});

browserModal.addEventListener('click', (e) => {
    if (e.target === browserModal) {
        closeFileBrowser();
    }
});

// Image popup events
popupClose.addEventListener('click', closeImagePopup);
popupImage.addEventListener('click', handlePopupImageClick);
popupImage2.addEventListener('click', handlePopupImageClick);
popupToggleBtn.addEventListener('click', handlePopupToggle);

document.querySelector('.popup-overlay').addEventListener('click', closeImagePopup);

// Keyboard events
document.addEventListener('keydown', (e) => {
    if (e.key === 'Escape') {
        if (!imagePopup.classList.contains('hidden')) {
            closeImagePopup();
        } else if (!applyModal.classList.contains('hidden')) {
            closeApplyModal();
        } else if (!browserModal.classList.contains('hidden')) {
            closeFileBrowser();
        }
    }
});

// Auto-save inputs on change
[folder1Input, folder2Input, folder3Input, labelFileInput].forEach(input => {
    input.addEventListener('change', saveConfig);
});

// Initialize
loadConfig();
