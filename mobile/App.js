import { useEffect, useRef, useState } from 'react';
import {
  ActivityIndicator, Image, Modal, Pressable, SafeAreaView, ScrollView,
  StatusBar, StyleSheet, Text, TextInput, View,
} from 'react-native';
import * as ImagePicker from 'expo-image-picker';
import { CameraView, useCameraPermissions } from 'expo-camera';
import AsyncStorage from '@react-native-async-storage/async-storage';
import { Ionicons } from '@expo/vector-icons';

// Sustituye esta IP por la IP local de la máquina donde corra FastAPI.
// También se puede cambiar desde la app: botón '⚙️ Configurar API'.
const DEFAULT_API_URL = 'http://192.168.1.100:8000';
const URL_STORAGE_KEY = 'egg_quality_api_url';
const HISTORY_STORAGE_KEY = 'egg_quality_history';

export default function App() {
  const [image, setImage] = useState(null);
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  const [apiUrl, setApiUrl] = useState(DEFAULT_API_URL);
  const [configOpen, setConfigOpen] = useState(false);
  const [configDraft, setConfigDraft] = useState(DEFAULT_API_URL);
  const [history, setHistory] = useState([]);
  const [historyOpen, setHistoryOpen] = useState(false);

  const [liveOpen, setLiveOpen] = useState(false);
  const [cameraPermission, requestCameraPermission] = useCameraPermissions();
  const cameraRef = useRef(null);

  // Cargar URL guardada e historial al iniciar.
  useEffect(() => {
    (async () => {
      const savedUrl = await AsyncStorage.getItem(URL_STORAGE_KEY);
      if (savedUrl) setApiUrl(savedUrl);
      const savedHistory = await AsyncStorage.getItem(HISTORY_STORAGE_KEY);
      if (savedHistory) setHistory(JSON.parse(savedHistory));
    })();
  }, []);

  async function chooseImage(source) {
    setError(null);
    setResult(null);
    const options = { mediaTypes: ['images'], quality: 0.9, allowsEditing: true, aspect: [1, 1] };
    const response = source === 'camera'
      ? await ImagePicker.launchCameraAsync(options)
      : await ImagePicker.launchImageLibraryAsync(options);
    if (!response.canceled) setImage(response.assets[0]);
  }

  // Captura el fotograma actual del preview en vivo y lo deja listo para analizar.
  async function takeLivePhoto(cameraRef) {
    setError(null);
    setResult(null);
    if (!cameraRef.current) return;
    const photo = await cameraRef.current.takePictureAsync({ quality: 0.9 });
    setImage({ uri: photo.uri, width: photo.width, height: photo.height });
    setLiveOpen(false);
  }

  async function saveApiUrl() {
    const url = configDraft.trim().replace(/\/+$/, '');
    if (!/^https?:\/\//.test(url)) {
      setError('La URL debe comenzar con http:// o https://');
      return;
    }
    setApiUrl(url);
    await AsyncStorage.setItem(URL_STORAGE_KEY, url);
    setConfigOpen(false);
    setError(null);
  }

  async function pushHistory(entry) {
    const next = [entry, ...history].slice(0, 10);
    setHistory(next);
    await AsyncStorage.setItem(HISTORY_STORAGE_KEY, JSON.stringify(next));
  }

  async function classify() {
    if (!image) return;
    setLoading(true);
    setError(null);
    try {
      const body = new FormData();
      body.append('file', { uri: image.uri, name: 'egg.jpg', type: 'image/jpeg' });
      const response = await fetch(`${apiUrl}/predict`, { method: 'POST', body });
      const data = await response.json();
      if (!response.ok) throw new Error(data.detail || 'No se pudo analizar la imagen.');
      setResult(data);
      pushHistory({
        label: data.label,
        confidence: data.confidence,
        at: new Date().toLocaleString(),
      });
    } catch (requestError) {
      setError(requestError.message);
    } finally {
      setLoading(false);
    }
  }

  const resultIsCrack = result?.label === 'crack';
  // Explicación breve: la app NO da un diagnóstico definitivo.
  const explanation = resultIsCrack
    ? 'Posible grieta detectada en la imagen. Se recomienda separar este huevo para una revisión manual.'
    : 'Sin grietas aparentes en la imagen. Confirma visualmente antes de tomar una decisión.';

  return (
    <SafeAreaView style={styles.safe}>
      <StatusBar barStyle="dark-content" />
      <ScrollView contentContainerStyle={styles.container}>
        <View style={styles.header}>
          <View style={styles.mark}><Ionicons name="egg" size={24} color="#F4EBDD" /></View>
          <View style={{ flex: 1 }}><Text style={styles.kicker}>CONTROL DE CALIDAD</Text><Text style={styles.title}>Egg Quality</Text></View>
          <Pressable onPress={() => setHistoryOpen(true)} style={styles.iconButton}><Ionicons name="list-outline" size={22} color="#29352D" /></Pressable>
          <Pressable onPress={() => { setConfigDraft(apiUrl); setConfigOpen(true); }} style={styles.iconButton}><Ionicons name="settings-outline" size={22} color="#29352D" /></Pressable>
        </View>
        <Text style={styles.intro}>Revisa un huevo en segundos.</Text>
        <Text style={styles.subtitle}>La cámara busca señales visibles de grieta y devuelve una estimación.</Text>

        {/* S4.1.1: instrucciones de encuadre siempre visibles */}
        <View style={styles.framingHint}>
          <Ionicons name="scan-outline" size={18} color="#B56A45" />
          <Text style={styles.framingText}>Encuadra UN solo huevo, centrado, bien iluminado y con la grieta visible si existe.</Text>
        </View>

        <View style={styles.preview}>
          {image ? <Image source={{ uri: image.uri }} style={styles.previewImage} /> : (
            <View style={styles.emptyPreview}><Ionicons name="scan-outline" size={48} color="#B56A45" /><Text style={styles.emptyTitle}>Aún no hay una imagen</Text><Text style={styles.emptyHint}>Usa una foto clara y bien iluminada</Text></View>
          )}
        </View>

        <View style={styles.actions}>
          <Pressable
            style={[styles.secondaryButton, styles.liveButton]}
            onPress={async () => {
              if (!cameraPermission?.granted) {
                const response = await requestCameraPermission();
                if (!response.granted) { setError('Se necesita permiso de cámara para el modo en vivo.'); return; }
              }
              setLiveOpen(true);
            }}
          >
            <Ionicons name="videocam-outline" size={20} color="#B56A45" />
            <Text style={[styles.secondaryText, styles.liveText]}>En vivo</Text>
          </Pressable>
          <Pressable style={styles.secondaryButton} onPress={() => chooseImage('camera')}><Ionicons name="camera-outline" size={20} color="#29352D" /><Text style={styles.secondaryText}>Foto</Text></Pressable>
          <Pressable style={styles.secondaryButton} onPress={() => chooseImage('library')}><Ionicons name="images-outline" size={20} color="#29352D" /><Text style={styles.secondaryText}>Galería</Text></Pressable>
        </View>

        <Pressable disabled={!image || loading} style={[styles.primaryButton, (!image || loading) && styles.disabled]} onPress={classify}>
          {loading ? <ActivityIndicator color="#F4EBDD" /> : <><Ionicons name="sparkles-outline" size={20} color="#F4EBDD" /><Text style={styles.primaryText}>Analizar huevo</Text></>}
        </Pressable>

        {error && <View style={styles.error}><Ionicons name="alert-circle-outline" size={20} color="#9E3F35" /><Text style={styles.errorText}>{error}</Text></View>}
        {result && (
          <View style={[styles.result, resultIsCrack ? styles.crackResult : styles.goodResult]}>
            <Ionicons name={resultIsCrack ? 'warning-outline' : 'checkmark-circle-outline'} size={30} color={resultIsCrack ? '#9E3F35' : '#3E654C'} />
            <View style={{ flex: 1 }}>
              <Text style={styles.resultLabel}>{resultIsCrack ? 'Posible grieta' : 'Aparenta estar bueno'}</Text>
              <Text style={styles.resultMeta}>Confianza: {(result.confidence * 100).toFixed(1)}%</Text>
              {/* S4.1.4: explicación breve, sin diagnóstico definitivo */}
              <Text style={styles.resultExplanation}>{explanation}</Text>
            </View>
          </View>
        )}

        {/* S4.1.2: modal de configuración de URL */}
        <Modal visible={configOpen} transparent animationType="slide">
          <View style={styles.modalBackdrop}>
            <View style={styles.modalCard}>
              <Text style={styles.modalTitle}>Configurar API</Text>
              <Text style={styles.modalSubtitle}>IP o dominio donde corre FastAPI (puerto incluido).</Text>
              <TextInput
                style={styles.input}
                value={configDraft}
                onChangeText={setConfigDraft}
                placeholder="http://192.168.1.100:8000"
                autoCapitalize="none"
                autoCorrect={false}
                keyboardType="url"
              />
              <Pressable style={styles.modalPrimary} onPress={saveApiUrl}><Text style={styles.modalPrimaryText}>Guardar</Text></Pressable>
              <Pressable style={styles.modalSecondary} onPress={() => setConfigOpen(false)}><Text style={styles.modalSecondaryText}>Cancelar</Text></Pressable>
            </View>
          </View>
        </Modal>

        {/* S4.1.3: historial local */}
        <Modal visible={historyOpen} transparent animationType="slide">
          <View style={styles.modalBackdrop}>
            <View style={styles.modalCard}>
              <Text style={styles.modalTitle}>Historial</Text>
              {history.length === 0 ? (
                <Text style={styles.modalSubtitle}>Aún no hay análisis guardados.</Text>
              ) : history.map((item, index) => (
                <View key={index} style={styles.historyItem}>
                  <Ionicons name={item.label === 'crack' ? 'warning-outline' : 'checkmark-circle-outline'} size={18} color={item.label === 'crack' ? '#9E3F35' : '#3E654C'} />
                  <Text style={styles.historyText}>
                    {item.label === 'crack' ? 'Posible grieta' : 'Buen estado'} · {(item.confidence * 100).toFixed(1)}%
                  </Text>
                  <Text style={styles.historyTime}>{item.at}</Text>
                </View>
              ))}
              <Pressable style={styles.modalSecondary} onPress={() => setHistoryOpen(false)}><Text style={styles.modalSecondaryText}>Cerrar</Text></Pressable>
            </View>
          </View>
        </Modal>

        {/* Cámara en vivo: preview continuo + captura manual del fotograma actual */}
        <Modal visible={liveOpen} animationType="slide">
          <View style={styles.liveContainer}>
            <CameraView ref={cameraRef} style={styles.camera} facing="back">
              <View style={styles.liveTop}>
                <Text style={styles.liveHint}>Encuadra UN huevo y toca el botón para capturar</Text>
              </View>
              <View style={styles.liveControls}>
                <Pressable style={styles.liveClose} onPress={() => setLiveOpen(false)}>
                  <Ionicons name="close" size={26} color="#F4EBDD" />
                </Pressable>
                <Pressable style={styles.captureButton} onPress={() => takeLivePhoto(cameraRef)}>
                  <View style={styles.captureInner} />
                </Pressable>
                <View style={styles.liveClose} />
              </View>
            </CameraView>
          </View>
        </Modal>

        <Text style={styles.footer}>API: {apiUrl}</Text>
      </ScrollView>
    </SafeAreaView>
  );
}

const styles = StyleSheet.create({
  safe: { flex: 1, backgroundColor: '#F4EBDD' },
  container: { padding: 24, paddingTop: 32, paddingBottom: 48 },
  header: { flexDirection: 'row', alignItems: 'center', gap: 12 },
  mark: { width: 48, height: 48, borderRadius: 16, backgroundColor: '#B56A45', alignItems: 'center', justifyContent: 'center' },
  kicker: { color: '#B56A45', fontSize: 11, fontWeight: '800', letterSpacing: 1.5 },
  title: { color: '#29352D', fontSize: 25, fontWeight: '800' },
  iconButton: { width: 40, height: 40, borderRadius: 12, backgroundColor: '#FFF9F0', alignItems: 'center', justifyContent: 'center' },
  intro: { marginTop: 30, color: '#29352D', fontSize: 32, fontWeight: '800', letterSpacing: -0.5 },
  subtitle: { marginTop: 8, color: '#6E756D', fontSize: 16, lineHeight: 23 },
  framingHint: { marginTop: 18, padding: 13, borderRadius: 14, backgroundColor: '#F1E0CC', flexDirection: 'row', gap: 9, alignItems: 'center' },
  framingText: { color: '#5C4632', fontSize: 14, lineHeight: 20, flex: 1 },
  preview: { marginTop: 18, height: 265, borderRadius: 24, overflow: 'hidden', backgroundColor: '#E8D9C7', borderWidth: 1, borderColor: '#D7C5B1' },
  previewImage: { width: '100%', height: '100%' },
  emptyPreview: { flex: 1, alignItems: 'center', justifyContent: 'center' },
  emptyTitle: { marginTop: 12, color: '#29352D', fontSize: 17, fontWeight: '700' },
  emptyHint: { marginTop: 5, color: '#7C817A', fontSize: 13 },
  actions: { flexDirection: 'row', gap: 12, marginTop: 16 },
  secondaryButton: { flex: 1, height: 48, borderRadius: 14, backgroundColor: '#FFF9F0', flexDirection: 'row', alignItems: 'center', justifyContent: 'center', gap: 8 },
  secondaryText: { color: '#29352D', fontWeight: '700', fontSize: 15 },
  primaryButton: { height: 54, borderRadius: 16, marginTop: 14, backgroundColor: '#29352D', alignItems: 'center', justifyContent: 'center', flexDirection: 'row', gap: 9 },
  primaryText: { color: '#F4EBDD', fontSize: 16, fontWeight: '800' },
  disabled: { opacity: 0.45 },
  error: { marginTop: 14, padding: 14, borderRadius: 14, backgroundColor: '#F6D8D2', flexDirection: 'row', gap: 8, alignItems: 'center' },
  errorText: { color: '#9E3F35', flex: 1 },
  result: { marginTop: 14, padding: 18, borderRadius: 18, flexDirection: 'row', gap: 12, alignItems: 'center' },
  goodResult: { backgroundColor: '#DCE9DD' },
  crackResult: { backgroundColor: '#F6D8D2' },
  resultLabel: { color: '#29352D', fontSize: 18, fontWeight: '800' },
  resultMeta: { marginTop: 3, color: '#6E756D', fontSize: 14 },
  resultExplanation: { marginTop: 6, color: '#4A4F48', fontSize: 13, lineHeight: 19 },
  modalBackdrop: { flex: 1, backgroundColor: 'rgba(41,53,45,0.45)', justifyContent: 'center', padding: 24 },
  modalCard: { backgroundColor: '#F4EBDD', borderRadius: 22, padding: 22 },
  modalTitle: { color: '#29352D', fontSize: 20, fontWeight: '800' },
  modalSubtitle: { marginTop: 6, color: '#6E756D', fontSize: 14, lineHeight: 20 },
  input: { marginTop: 14, height: 48, borderRadius: 12, backgroundColor: '#FFF9F0', paddingHorizontal: 14, color: '#29352D', fontSize: 15, borderWidth: 1, borderColor: '#D7C5B1' },
  modalPrimary: { marginTop: 16, height: 50, borderRadius: 14, backgroundColor: '#29352D', alignItems: 'center', justifyContent: 'center' },
  modalPrimaryText: { color: '#F4EBDD', fontSize: 15, fontWeight: '800' },
  modalSecondary: { marginTop: 10, height: 46, borderRadius: 14, backgroundColor: '#E8D9C7', alignItems: 'center', justifyContent: 'center' },
  modalSecondaryText: { color: '#29352D', fontSize: 15, fontWeight: '700' },
  historyItem: { marginTop: 10, backgroundColor: '#FFF9F0', borderRadius: 12, padding: 12, flexDirection: 'row', gap: 8, alignItems: 'center' },
  historyText: { color: '#29352D', fontSize: 14, fontWeight: '700', flex: 1 },
  historyTime: { color: '#7C817A', fontSize: 12 },
  footer: { marginTop: 20, color: '#9A9F98', fontSize: 12, textAlign: 'center' },
  liveButton: { backgroundColor: '#F1E0CC', borderWidth: 1, borderColor: '#E0C4A8' },
  liveText: { color: '#B56A45' },
  liveContainer: { flex: 1, backgroundColor: '#29352D' },
  camera: { flex: 1, justifyContent: 'space-between' },
  liveTop: { paddingTop: 56, paddingHorizontal: 24, alignItems: 'center' },
  liveHint: { color: '#F4EBDD', fontSize: 14, fontWeight: '700', backgroundColor: 'rgba(41,53,45,0.55)', paddingHorizontal: 14, paddingVertical: 8, borderRadius: 12, overflow: 'hidden', textAlign: 'center' },
  liveControls: { flexDirection: 'row', alignItems: 'center', justifyContent: 'space-between', paddingHorizontal: 36, paddingBottom: 40 },
  liveClose: { width: 48, height: 48, borderRadius: 24, backgroundColor: 'rgba(41,53,45,0.55)', alignItems: 'center', justifyContent: 'center' },
  captureButton: { width: 76, height: 76, borderRadius: 38, backgroundColor: 'rgba(244,235,221,0.35)', alignItems: 'center', justifyContent: 'center' },
  captureInner: { width: 58, height: 58, borderRadius: 29, backgroundColor: '#F4EBDD' },
});