(() => {
  const LS_PREFIX = 'roadmap-v1-';
  // Full roadmap items (model, project, short why, dataset link, starter hint)
  const ROADMAP = {
    stage1: [
      {id:"s1-1", model:"Linear Regression", project:"House Price Prediction 🏡",
       why:"Predict continuous targets (prices). Good first regression project.",
       dataset:"Kaggle: House Prices (https://www.kaggle.com/c/house-prices-advanced-regression-techniques)",
       hint:"from sklearn.linear_model import LinearRegression\nmodel = LinearRegression().fit(X_train, y_train)"},
      {id:"s1-2", model:"Logistic Regression", project:"Titanic Survival Prediction 🚢",
       why:"Binary classification baseline; interpretable coefficients.",
       dataset:"Kaggle: Titanic (https://www.kaggle.com/c/titanic)",
       hint:"from sklearn.linear_model import LogisticRegression\nmodel.fit(X_train, y_train)"},
      {id:"s1-3", model:"Naive Bayes", project:"Spam Email Classifier 📧",
       why:"Simple, fast for text (bag-of-words) classification.",
       dataset:"Enron / Kaggle Spam datasets",
       hint:"from sklearn.naive_bayes import MultinomialNB\nclf = MultinomialNB().fit(X_train_vec, y_train)"},
      {id:"s1-4", model:"k-NN (K-Nearest Neighbors)", project:"Iris Classification 🌸",
       why:"Instance-based method; good for intuition about distance metrics.",
       dataset:"sklearn.datasets: Iris",
       hint:"from sklearn.neighbors import KNeighborsClassifier\nKNeighborsClassifier(3).fit(X,y)"},
      {id:"s1-5", model:"SVM (Support Vector Machine)", project:"Handwritten Digit Recognition ✍️",
       why:"Strong for medium-sized, high-dim classification problems.",
       dataset:"MNIST (sklearn or tensorflow datasets)",
       hint:"from sklearn.svm import SVC\nSVC(kernel='rbf').fit(X_train, y_train)"}
    ],
    stage2: [
      {id:"s2-1", model:"Decision Tree", project:"Loan Approval Prediction 💳",
       why:"Interpretable rule-based classifier; visualize splits.",
       dataset:"Loan datasets (Kaggle)",
       hint:"from sklearn.tree import DecisionTreeClassifier\nDecisionTreeClassifier().fit(X,y)"},
      {id:"s2-2", model:"Random Forest", project:"Customer Churn Prediction 📞",
       why:"Bagging ensemble to reduce overfitting; feature importance.",
       dataset:"Telco Churn dataset (Kaggle)",
       hint:"from sklearn.ensemble import RandomForestClassifier\nRandomForestClassifier(n_estimators=100).fit(X,y)"},
      {id:"s2-3", model:"Gradient Boosting (XGBoost/LightGBM)", project:"Fraud Detection 💰",
       why:"Powerful for structured/tabular data, competitive in contests.",
       dataset:"Credit Card Fraud (Kaggle)",
       hint:"import xgboost as xgb\nxgb.XGBClassifier().fit(X,y)"}
    ],
    stage3: [
      {id:"s3-1", model:"K-Means Clustering", project:"Customer Segmentation 🛒",
       why:"Unsupervised grouping; use elbow/silhouette to pick k.",
       dataset:"Mall customers / retail datasets",
       hint:"from sklearn.cluster import KMeans\nKMeans(n_clusters=4).fit(X)"},
      {id:"s3-2", model:"PCA (Principal Component Analysis)", project:"Dimensionality Reduction & Visualization 📉",
       why:"Project high-dim data to low-dim for visualization & speedups.",
       dataset:"High-dim datasets (gene expression etc.)",
       hint:"from sklearn.decomposition import PCA\nPCA(n_components=2).fit_transform(X)"},
      {id:"s3-3", model:"DBSCAN", project:"Anomaly Detection / Density Clustering 🚨",
       why:"Detect noise/outliers and discover dense regions without k.",
       dataset:"Network traffic / logs",
       hint:"from sklearn.cluster import DBSCAN\nDBSCAN(eps=0.5).fit(X)"},
      {id:"s3-4", model:"Gaussian Mixture Models (GMM)", project:"Soft Clustering 🛍️",
       why:"Probabilistic clusters, useful when clusters overlap.",
       dataset:"Customer data / gaussian-like clusters",
       hint:"from sklearn.mixture import GaussianMixture\nGaussianMixture(n_components=3).fit(X)"},
      {id:"s3-5", model:"ARIMA / Prophet", project:"Time Series Forecasting 📈",
       why:"Classical & modern tools for forecasting temporal data.",
       dataset:"Stock / sales / energy time series",
       hint:"statsmodels.tsa.arima_model.ARIMA or Prophet from Facebook"}
    ],
    stage4: [
      {id:"s4-1", model:"Feedforward Neural Network (MLP)", project:"Tabular NN Baseline 📊",
       why:"Basic neural net for tabular data — good DL introduction.",
       dataset:"Any tabular dataset",
       hint:"keras.Sequential([Dense(...),...])"},
      {id:"s4-2", model:"CNN (Convolutional Neural Network)", project:"Image Classification (MNIST / CIFAR) 🖼️",
       why:"Learn spatial filters, transfer learning & data augmentation.",
       dataset:"MNIST, CIFAR-10 (Keras datasets / Kaggle)",
       hint:"tf.keras.layers.Conv2D(...)"},
      {id:"s4-3", model:"RNN / LSTM", project:"Sentiment Analysis on Reviews 🎬",
       why:"Sequence modeling for text/time series; learn embeddings.",
       dataset:"IMDB Reviews",
       hint:"keras.layers.LSTM(...)\nuse Embedding layer"},
      {id:"s4-4", model:"Transformers (BERT/GPT)", project:"Text Summarization / QA 📰",
       why:"State-of-the-art for many NLP tasks via self-attention.",
       dataset:"CNN/DailyMail, SQuAD, Hugging Face datasets",
       hint:"from transformers import AutoModelForSeq2SeqLM, AutoTokenizer"},
      {id:"s4-5", model:"Autoencoder / VAE", project:"Compression & Anomaly Detection 🧩",
       why:"Learn latent representations; use reconstruction error for anomalies.",
       dataset:"Images or sensor data",
       hint:"Build encoder/decoder in Keras/PyTorch"}
    ],
    stage5: [
      {id:"s5-1", model:"GANs (Generative Adversarial Networks)", project:"Generate Faces / Images 👤",
       why:"Generative modeling; adversarial training dynamics.",
       dataset:"CelebA, FFHQ",
       hint:"Build generator & discriminator and train adversarially"},
      {id:"s5-2", model:"Reinforcement Learning (DQN / PPO)", project:"Game Agent (CartPole) 🎮",
       why:"Decision-making under rewards; agent-environment loop.",
       dataset:"OpenAI Gym environments",
       hint:"Use stable-baselines3 or custom RL loop"},
      {id:"s5-3", model:"Graph Neural Networks (GNNs)", project:"Link Prediction / Node Classification 🌐",
       why:"Work with graph-structured data (social, molecules).",
       dataset:"Cora / PubMed / custom graphs",
       hint:"Use PyTorch Geometric or DGL"},
      {id:"s5-4", model:"Deployment & MLOps", project:"Serve Models + Monitor 🚀",
       why:"Make ML usable in production: APIs, monitoring, versioning.",
       dataset:"Your model artifacts",
       hint:"FastAPI + Docker + CI/CD; TF Serving or TorchServe"}
    ]
  };

  // DOM caches
  const containers = {
    stage1: document.getElementById('list-stage-1'),
    stage2: document.getElementById('list-stage-2'),
    stage3: document.getElementById('list-stage-3'),
    stage4: document.getElementById('list-stage-4'),
    stage5: document.getElementById('list-stage-5')
  };

  // render a single card
  function makeCard(item){
    const card = document.createElement('article');
    card.className = 'card';
    card.dataset.id = item.id;

    const row = document.createElement('div');
    row.className = 'card-row';

    const cbWrap = document.createElement('div');
    cbWrap.className = 'cb-wrap';
    const cb = document.createElement('input');
    cb.type = 'checkbox';
    cb.id = 'cb-' + item.id;
    cb.title = 'Mark project done';
    cbWrap.appendChild(cb);

    const meta = document.createElement('div');
    meta.className = 'meta';
    meta.innerHTML = `
      <div class="model">${item.model}</div>
      <div class="project">${item.project}</div>
      <div class="why"><strong>Why:</strong> ${item.why || ''}</div>
      <div class="dataset"><strong>Dataset / sources:</strong> ${item.dataset || 'Search Kaggle / UCI / HuggingFace'}</div>
      <div class="actions">
        <button class="btn expandBtn" aria-expanded="false">Details</button>
        <button class="btn copyBtn ghost" title="Copy starter code">Copy code</button>
      </div>
      <div class="expandable">
        <div class="hint"><strong>Starter hint:</strong><br><pre style="margin:0;white-space:pre-wrap;">${escapeHtml(item.hint || '')}</pre></div>
        <div class="small">Tip: Try baseline → evaluate → improve (features → tuning → ensembles/DL).</div>
      </div>
    `;

    row.appendChild(cbWrap);
    row.appendChild(meta);
    card.appendChild(row);

    // restore state
    const stored = localStorage.getItem(LS_PREFIX + item.id);
    if (stored === '1') {
      cb.checked = true;
      card.classList.add('done');
    }

    // checkbox handler
    cb.addEventListener('change', () => {
      if (cb.checked) {
        localStorage.setItem(LS_PREFIX + item.id, '1');
        card.classList.add('done');
      } else {
        localStorage.removeItem(LS_PREFIX + item.id);
        card.classList.remove('done');
      }
      updateProgress();
    });

    // expand/collapse details
    const expandBtn = meta.querySelector('.expandBtn');
    const expandable = meta.querySelector('.expandable');
    expandBtn.addEventListener('click', () => {
      const exp = card.classList.toggle('expanded');
      expandBtn.textContent = exp ? 'Hide' : 'Details';
      expandBtn.setAttribute('aria-expanded', exp ? 'true' : 'false');
    });

    // copy starter code
    const copyBtn = meta.querySelector('.copyBtn');
    copyBtn.addEventListener('click', () => {
      const text = item.hint || '';
      navigator.clipboard?.writeText(text).then(()=> {
        copyBtn.textContent = 'Copied ✓';
        setTimeout(()=> copyBtn.textContent = 'Copy code', 1500);
      }).catch(()=> {
        alert('Copy failed — select and copy manually.');
      });
    });

    return card;
  }

  // escape simple HTML for hints
  function escapeHtml(s){
    return s.replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;');
  }

  // render all stages
  function renderAll(){
    containers.stage1.innerHTML = '';
    containers.stage2.innerHTML = '';
    containers.stage3.innerHTML = '';
    containers.stage4.innerHTML = '';
    containers.stage5.innerHTML = '';

    ROADMAP.stage1.forEach(it => containers.stage1.appendChild(makeCard(it)));
    ROADMAP.stage2.forEach(it => containers.stage2.appendChild(makeCard(it)));
    ROADMAP.stage3.forEach(it => containers.stage3.appendChild(makeCard(it)));
    ROADMAP.stage4.forEach(it => containers.stage4.appendChild(makeCard(it)));
    ROADMAP.stage5.forEach(it => containers.stage5.appendChild(makeCard(it)));

    updateProgress();
  }

  // update progress bar
  function updateProgress(){
    const allBoxes = Array.from(document.querySelectorAll('.card input[type="checkbox"]'));
    const total = allBoxes.length;
    const done = allBoxes.filter(b => b.checked).length;
    const pct = total ? Math.round((done/total)*100) : 0;
    document.getElementById('progressFill').style.width = pct + '%';
    document.getElementById('progressPercent').textContent = pct + '%';
  }

  // search filter
  const searchEl = document.getElementById('search');
  searchEl.addEventListener('input', () => {
    const q = searchEl.value.trim().toLowerCase();
    document.querySelectorAll('.card').forEach(card => {
      const text = (card.textContent || '').toLowerCase();
      card.style.display = q ? (text.includes(q) ? '' : 'none') : '';
    });
  });

  // export/import/reset
  document.getElementById('exportBtn').addEventListener('click', () => {
    const data = {};
    document.querySelectorAll('.card').forEach(card => {
      const id = card.dataset.id;
      data[id] = localStorage.getItem(LS_PREFIX + id) === '1';
    });
    const blob = new Blob([JSON.stringify({progress:data},null,2)], {type:'application/json'});
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a'); a.href = url; a.download = 'roadmap-progress.json'; document.body.appendChild(a); a.click(); a.remove();
    URL.revokeObjectURL(url);
  });

  const importFile = document.getElementById('importFile');
  document.getElementById('importBtn').addEventListener('click', () => importFile.click());
  importFile.addEventListener('change', (e) => {
    const f = e.target.files[0];
    if (!f) return;
    const reader = new FileReader();
    reader.onload = () => {
      try {
        const parsed = JSON.parse(reader.result);
        if (!parsed || !parsed.progress) { alert('Invalid file'); return; }
        Object.entries(parsed.progress).forEach(([id,val]) => {
          if (val) localStorage.setItem(LS_PREFIX + id, '1'); else localStorage.removeItem(LS_PREFIX + id);
        });
        renderAll();
        alert('Progress imported.');
      } catch (err) { alert('Could not parse file'); }
    };
    reader.readAsText(f);
    importFile.value = '';
  });

  document.getElementById('resetBtn').addEventListener('click', () => {
    if (!confirm('Reset all progress?')) return;
    document.querySelectorAll('.card').forEach(card => {
      const id = card.dataset.id;
      localStorage.removeItem(LS_PREFIX + id);
    });
    renderAll();
  });

  // keyboard focus: press '/'
  document.addEventListener('keydown', (e) => {
    if (e.key === '/' && document.activeElement.tagName !== 'INPUT') {
      e.preventDefault();
      searchEl.focus();
    }
  });

  // initial render & storage sync
  renderAll();
  window.addEventListener('storage', renderAll);

})();